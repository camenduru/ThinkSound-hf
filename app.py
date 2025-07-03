from prefigure.prefigure import get_all_args, push_wandb_config
import spaces
import json
import os
os.environ["GRADIO_TEMP_DIR"] = "./.gradio_tmp"
import re
import torch
import torchaudio
# import pytorch_lightning as pl
import lightning as L
from lightning.pytorch.callbacks import Timer, ModelCheckpoint, BasePredictionWriter
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.tuner import Tuner
from lightning.pytorch import seed_everything
import random
from datetime import datetime
from ThinkSound.data.datamodule import DataModule
from ThinkSound.models import create_model_from_config
from ThinkSound.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from ThinkSound.training import create_training_wrapper_from_config, create_demo_callback_from_config
from ThinkSound.training.utils import copy_state_dict
from ThinkSound.inference.sampling import get_alphas_sigmas, sample, sample_discrete_euler
from data_utils.v2a_utils.feature_utils_224 import FeaturesUtils
from torch.utils.data import Dataset
from typing import Optional, Union
from torchvision.transforms import v2
from torio.io import StreamingMediaDecoder
from torchvision.utils import save_image
from transformers import AutoProcessor
import torch.nn.functional as F
import gradio as gr
import tempfile
import subprocess
from huggingface_hub import hf_hub_download
from moviepy.editor import VideoFileClip
# os.system("conda install -c conda-forge 'ffmpeg<7'")

_CLIP_SIZE = 224
_CLIP_FPS = 8.0

_SYNC_SIZE = 224
_SYNC_FPS = 25.0

def pad_to_square(video_tensor):
    if len(video_tensor.shape) != 4:
        raise ValueError("Input tensor must have shape (l, c, h, w)")

    l, c, h, w = video_tensor.shape
    max_side = max(h, w)

    pad_h = max_side - h
    pad_w = max_side - w
    
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)

    video_padded = F.pad(video_tensor, pad=padding, mode='constant', value=0)

    return video_padded


class VGGSound(Dataset):

    def __init__(
        self,
        sample_rate: int = 44_100,
        duration_sec: float = 9.0,
        audio_samples: int = None,
        normalize_audio: bool = False,
    ):
        if audio_samples is None:
            self.audio_samples = int(sample_rate * duration_sec)
        else:
            self.audio_samples = audio_samples
            effective_duration = audio_samples / sample_rate
            # make sure the duration is close enough, within 15ms
            assert abs(effective_duration - duration_sec) < 0.015, \
                f'audio_samples {audio_samples} does not match duration_sec {duration_sec}'

        self.sample_rate = sample_rate
        self.duration_sec = duration_sec

        self.expected_audio_length = self.audio_samples
        self.clip_expected_length = int(_CLIP_FPS * self.duration_sec)
        self.sync_expected_length = int(_SYNC_FPS * self.duration_sec)

        self.clip_transform = v2.Compose([
            v2.Lambda(pad_to_square),          # 先填充为正方形
            v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        self.clip_processor = AutoProcessor.from_pretrained("facebook/metaclip-h14-fullcc2.5b")
        self.sync_transform = v2.Compose([
            v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(_SYNC_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.resampler = {}

    def sample(self, video_path,label,cot):
        video_id = video_path

        reader = StreamingMediaDecoder(video_path)
        reader.add_basic_video_stream(
            frames_per_chunk=int(_CLIP_FPS * self.duration_sec),
            frame_rate=_CLIP_FPS,
            format='rgb24',
        )
        reader.add_basic_video_stream(
            frames_per_chunk=int(_SYNC_FPS * self.duration_sec),
            frame_rate=_SYNC_FPS,
            format='rgb24',
        ) 

        reader.fill_buffer()
        data_chunk = reader.pop_chunks()

        clip_chunk = data_chunk[0]
        sync_chunk = data_chunk[1]

        if sync_chunk is None:
            raise RuntimeError(f'Sync video returned None {video_id}')

        clip_chunk = clip_chunk[:self.clip_expected_length]
        # import ipdb
        # ipdb.set_trace()
        if clip_chunk.shape[0] != self.clip_expected_length:
            current_length = clip_chunk.shape[0]
            padding_needed = self.clip_expected_length - current_length
            
            # Check that padding needed is no more than 2
            assert padding_needed < 4, f'Padding no more than 2 frames allowed, but {padding_needed} needed'

            # If assertion passes, proceed with padding
            if padding_needed > 0:
                last_frame = clip_chunk[-1]
                log.info(last_frame.shape) 
                # Repeat the last frame to reach the expected length
                padding = last_frame.repeat(padding_needed, 1, 1, 1)
                clip_chunk = torch.cat((clip_chunk, padding), dim=0)
            # raise RuntimeError(f'CLIP video wrong length {video_id}, '
            #                    f'expected {self.clip_expected_length}, '
            #                    f'got {clip_chunk.shape[0]}')
        
        # save_image(clip_chunk[0] / 255.0,'ori.png')
        clip_chunk = pad_to_square(clip_chunk)

        clip_chunk = self.clip_processor(images=clip_chunk, return_tensors="pt")["pixel_values"]

        sync_chunk = sync_chunk[:self.sync_expected_length]
        if sync_chunk.shape[0] != self.sync_expected_length:
            # padding using the last frame, but no more than 2
            current_length = sync_chunk.shape[0]
            last_frame = sync_chunk[-1]

            padding = last_frame.repeat(self.sync_expected_length - current_length, 1, 1, 1)
            assert self.sync_expected_length - current_length < 12, f'sync can pad no more than 2 while {self.sync_expected_length - current_length}'
            sync_chunk = torch.cat((sync_chunk, padding), dim=0)
            # raise RuntimeError(f'Sync video wrong length {video_id}, '
            #                    f'expected {self.sync_expected_length}, '
            #                    f'got {sync_chunk.shape[0]}')
        
        sync_chunk = self.sync_transform(sync_chunk)
        # assert audio_chunk.shape[1] == self.expected_audio_length and clip_chunk.shape[0] == self.clip_expected_length \
        # and sync_chunk.shape[0] == self.sync_expected_length, 'error processed data shape'
        data = {
            'id': video_id,
            'caption': label,
            'caption_cot': cot,
            # 'audio': audio_chunk,
            'clip_video': clip_chunk,
            'sync_video': sync_chunk,
        }

        return data

# 检查设备
if torch.cuda.is_available():
    device = 'cuda'
    extra_device = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0'
else:
    device = 'cpu'
    extra_device = 'cpu'

print(f"load in device {device}")

vae_ckpt = hf_hub_download(repo_id="FunAudioLLM/ThinkSound", filename="vae.ckpt",repo_type="model")
synchformer_ckpt = hf_hub_download(repo_id="FunAudioLLM/ThinkSound", filename="synchformer_state_dict.pth",repo_type="model")

feature_extractor = FeaturesUtils(
    vae_ckpt=None,
    vae_config='ThinkSound/configs/model_configs/stable_audio_2_0_vae.json',
    enable_conditions=True,
    synchformer_ckpt=synchformer_ckpt
).eval().to(extra_device)

args = get_all_args()

seed = 10086

seed_everything(seed, workers=True)


#Get JSON config from args.model_config
with open("ThinkSound/configs/model_configs/thinksound.json") as f:
    model_config = json.load(f)

model = create_model_from_config(model_config)

## speed by torch.compile
if args.compile:
    model = torch.compile(model)
    
if args.pretrained_ckpt_path:
    copy_state_dict(model, load_ckpt_state_dict(args.pretrained_ckpt_path,prefix='diffusion.')) # autoencoder.  diffusion.

if args.remove_pretransform_weight_norm == "pre_load":
    remove_weight_norm_from_model(model.pretransform)


load_vae_state = load_ckpt_state_dict(vae_ckpt, prefix='autoencoder.') 
# new_state_dict = {k.replace("autoencoder.", ""): v for k, v in load_vae_state.items() if k.startswith("autoencoder.")}
model.pretransform.load_state_dict(load_vae_state)

# Remove weight_norm from the pretransform if specified
if args.remove_pretransform_weight_norm == "post_load":
    remove_weight_norm_from_model(model.pretransform)
ckpt_path = hf_hub_download(repo_id="FunAudioLLM/ThinkSound", filename="thinksound.ckpt",repo_type="model")
training_wrapper = create_training_wrapper_from_config(model_config, model)
# 加载模型权重时根据设备选择map_location
training_wrapper.load_state_dict(torch.load(ckpt_path)['state_dict'])

training_wrapper.to("cuda")

def get_video_duration(video_path):
    video = VideoFileClip(video_path)
    return video.duration

@spaces.GPU(duration=60)
@torch.inference_mode()
@torch.no_grad()
def synthesize_video_with_audio(video_file, caption, cot):
    audio_path = get_audio(video_file, caption, cot)
    video_path = video_file
    if caption is None:
        caption = ''
    if cot is None:
        cot = caption
    timer = Timer(duration="00:15:00:00")
    #get video duration
    duration_sec = get_video_duration(video_path)
    print(duration_sec)
    preprocesser = VGGSound(duration_sec=duration_sec)
    data = preprocesser.sample(video_path, caption, cot)

    yield "⏳ Extracting Features…", None

    preprocessed_data = {}
    metaclip_global_text_features, metaclip_text_features = feature_extractor.encode_text(data['caption'])
    preprocessed_data['metaclip_global_text_features'] = metaclip_global_text_features.detach().cpu().squeeze(0)
    preprocessed_data['metaclip_text_features'] = metaclip_text_features.detach().cpu().squeeze(0)

    t5_features = feature_extractor.encode_t5_text(data['caption_cot'])
    preprocessed_data['t5_features'] = t5_features.detach().cpu().squeeze(0)

    clip_features = feature_extractor.encode_video_with_clip(data['clip_video'].unsqueeze(0).to(extra_device))
    preprocessed_data['metaclip_features'] = clip_features.detach().cpu().squeeze(0)

    sync_features = feature_extractor.encode_video_with_sync(data['sync_video'].unsqueeze(0).to(extra_device))
    preprocessed_data['sync_features'] = sync_features.detach().cpu().squeeze(0)
    preprocessed_data['video_exist'] = torch.tensor(True)
    print("clip_shape", preprocessed_data['metaclip_features'].shape)
    print("sync_shape", preprocessed_data['sync_features'].shape)
    sync_seq_len = preprocessed_data['sync_features'].shape[0]
    clip_seq_len = preprocessed_data['metaclip_features'].shape[0]
    latent_seq_len = (int)(194/9*duration_sec)
    training_wrapper.diffusion.model.model.update_seq_lengths(latent_seq_len, clip_seq_len, sync_seq_len)

    metadata = [preprocessed_data]

    batch_size = 1
    length = latent_seq_len
    with torch.amp.autocast(device):
        conditioning = training_wrapper.diffusion.conditioner(metadata, training_wrapper.device)
    
    video_exist = torch.stack([item['video_exist'] for item in metadata],dim=0)
    conditioning['metaclip_features'][~video_exist] = training_wrapper.diffusion.model.model.empty_clip_feat
    conditioning['sync_features'][~video_exist] = training_wrapper.diffusion.model.model.empty_sync_feat

    yield "⏳ Inferring…", None
    cond_inputs = training_wrapper.diffusion.get_conditioning_inputs(conditioning)
    noise = torch.randn([batch_size, training_wrapper.diffusion.io_channels, length]).to(training_wrapper.device)
    with torch.amp.autocast(device):
        model = training_wrapper.diffusion.model
        if training_wrapper.diffusion_objective == "v":
            fakes = sample(model, noise, 24, 0, **cond_inputs, cfg_scale=5, batch_cfg=True)
        elif training_wrapper.diffusion_objective == "rectified_flow":
            import time
            start_time = time.time()
            fakes = sample_discrete_euler(model, noise, 24, **cond_inputs, cfg_scale=5, batch_cfg=True)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"执行时间: {execution_time:.2f} 秒")
        if training_wrapper.diffusion.pretransform is not None:
            fakes = training_wrapper.diffusion.pretransform.decode(fakes)

    audios = fakes.to(torch.float32).div(torch.max(torch.abs(fakes))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
        torchaudio.save(tmp_audio.name, audios[0], 44100)
        audio_path = tmp_audio.name

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
        output_video_path = tmp_video.name

    cmd = [
        'ffmpeg', '-y', '-i', video_file, '-i', audio_path,
        '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0',
        '-shortest', output_video_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # return output_video_path
    yield "✅ Generation completed!", output_video_path

# Gradio界面
with gr.Blocks() as demo:
    gr.Markdown(
        """
# ThinkSound\n
ThinkSound is a unified Any2Audio generation framework with flow matching guided by Chain-of-Thought (CoT) reasoning.

Upload video and caption (optional), and get video with audio!  

"""
    )
    with gr.Row():
        video_input = gr.Video(label="upload video")
        with gr.Column():
            caption_input = gr.Textbox(label="caption(optional)", placeholder="can be empty", lines=1)
            cot_input = gr.Textbox(label="CoT Description (optional)", placeholder="can be empty", lines=6)
    output_video = gr.Video(label="output video")
    btn = gr.Button("start synthesize")
    btn.click(fn=synthesize_video_with_audio, inputs=[video_input, caption_input, cot_input], outputs=output_video)

    gr.Examples(
        examples=[
            ["./examples/3_mute.mp4", "Gentle Sucking Sounds From the Pacifier", "Begin by creating a soft, steady background of light pacifier suckling. Add subtle, breathy rhythms to mimic a newborn's gentle mouth movements. Keep the sound smooth, natural, and soothing.","./examples/3.mp4"],
            ["./examples/2_mute.mp4", "Printer Printing", "Generate a continuous printer printing sound with periodic beeps and paper movement, plus a cat pawing at the machine. Add subtle ambient room noise for authenticity, keeping the focus on printing, beeps, and the cat's interaction.", "./examples/2.mp4"],
            ["./examples/5_mute.mp4", "Lighting Firecrackers", "Generate the sound of firecrackers lighting and exploding repeatedly on the ground, followed by fireworks bursting in the sky. Incorporate occasional subtle echoes to mimic an outdoor night ambiance, with no human voices present.","./examples/5.mp4"],
            ["./examples/4_mute.mp4", "Plastic Debris Handling", "Begin with the sound of hands scooping up loose plastic debris, followed by the subtle cascading noise as the pieces fall and scatter back down. Include soft crinkling and rustling to emphasize the texture of the plastic. Add ambient factory background noise with distant machinery to create an industrial atmosphere.","./examples/4.mp4"]
        ],
        inputs=[video_input, caption_input, cot_input, output_video],
    )
    
demo.launch(share=True)



