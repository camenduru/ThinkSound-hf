from prefigure.prefigure import get_all_args, push_wandb_config
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
# from think_sound.data.dataset import create_dataloader_from_config
from think_sound.data.datamodule import DataModule
from think_sound.models import create_model_from_config
from think_sound.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from think_sound.training import create_training_wrapper_from_config, create_demo_callback_from_config
from think_sound.training.utils import copy_state_dict
from think_sound.inference.sampling import get_alphas_sigmas, sample, sample_discrete_euler
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
        audio_samples: Optional[int] = 397312,
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

    def sample(self, video_path,label):
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
            # 重复最后一帧以进行填充
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

vae_ckpt = hf_hub_download(repo_id="UncleWang233/occdata", filename="epoch=3-step=100000.ckpt",repo_type="dataset")
synchformer_ckpt = hf_hub_download(repo_id="UncleWang233/occdata", filename="synchformer_state_dict.pth",repo_type="dataset")
feature_extractor = FeaturesUtils(
    vae_ckpt=vae_ckpt,
    vae_config='think_sound/configs/model_configs/autoencoders/stable_audio_2_0_vae.json',
    enable_conditions=True,
    synchformer_ckpt=synchformer_ckpt
).eval().to(extra_device)

preprocesser = VGGSound()

args = get_all_args()

seed = 10086

seed_everything(seed, workers=True)


#Get JSON config from args.model_config
with open("think_sound/configs/model_configs/vt2audio/latent_clip_224_text_sync_mmdit_flow_logit_t5_kernel_size3.json") as f:
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
ckpt_path = hf_hub_download(repo_id="UncleWang233/occdata", filename="epoch=10-step=68000.ckpt",repo_type="dataset")
training_wrapper = create_training_wrapper_from_config(model_config, model)
# 加载模型权重时根据设备选择map_location
if device == 'cuda':
    training_wrapper.load_state_dict(torch.load(ckpt_path)['state_dict'])
else:
    training_wrapper.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict'])

def get_audio(video_path, caption):
    # 允许caption为空
    if caption is None:
        caption = ''
    timer = Timer(duration="00:15:00:00")
    data = preprocesser.sample(video_path, caption)

    preprocessed_data = {}
    metaclip_global_text_features, metaclip_text_features = feature_extractor.encode_text(data['caption'])
    preprocessed_data['metaclip_global_text_features'] = metaclip_global_text_features.detach().cpu().squeeze(0)
    preprocessed_data['metaclip_text_features'] = metaclip_text_features.detach().cpu().squeeze(0)

    t5_features = feature_extractor.encode_t5_text(data['caption'])
    preprocessed_data['t5_features'] = t5_features.detach().cpu().squeeze(0)

    clip_features = feature_extractor.encode_video_with_clip(data['clip_video'].unsqueeze(0).to(extra_device))
    preprocessed_data['metaclip_features'] = clip_features.detach().cpu().squeeze(0)

    sync_features = feature_extractor.encode_video_with_sync(data['sync_video'].unsqueeze(0).to(extra_device))
    preprocessed_data['sync_features'] = sync_features.detach().cpu().squeeze(0)
    preprocessed_data['video_exist'] = torch.tensor(True)

    metadata = [preprocessed_data]

    batch_size = 1
    length = 194
    with torch.amp.autocast(device):
        conditioning = training_wrapper.diffusion.conditioner(metadata, training_wrapper.device)
    
    video_exist = torch.stack([item['video_exist'] for item in metadata],dim=0)
    conditioning['metaclip_features'][~video_exist] = training_wrapper.diffusion.model.model.empty_clip_feat
    conditioning['sync_features'][~video_exist] = training_wrapper.diffusion.model.model.empty_sync_feat

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
    # 保存临时音频文件
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
        torchaudio.save(tmp_audio.name, audios[0], 44100)
        audio_path = tmp_audio.name
    return audio_path

# 合成新视频：用ffmpeg将音频与原视频合成

def synthesize_video_with_audio(video_file, caption):
    # 允许caption为空
    if caption is None:
        caption = ''
    audio_path = get_audio(video_file, caption)
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
        output_video_path = tmp_video.name
    # ffmpeg命令：用新音频替换原视频音轨
    cmd = [
        'ffmpeg', '-y', '-i', video_file, '-i', audio_path,
        '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0',
        '-shortest', output_video_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_video_path

# Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# ThinkSound\nupload video and caption(optional), and get video with audio!")
    with gr.Row():
        video_input = gr.Video(label="upload video")
        caption_input = gr.Textbox(label="caption(optional)", placeholder="can be empty", lines=1)
    output_video = gr.Video(label="output video")
    btn = gr.Button("start synthesize")
    btn.click(fn=synthesize_video_with_audio, inputs=[video_input, caption_input], outputs=output_video)

    gr.Examples(
        examples=[
            ["./examples/1_mute.mp4", "Playing Trumpet"],
            ["./examples/2_mute.mp4", "Axe striking"],
            ["./examples/3_mute.mp4", "Gentle Sucking Sounds From the Pacifier"],
            ["./examples/4_mute.mp4", "train passing by"],
            ["./examples/5_mute.mp4", "Lighting Firecrackers"]
        ],
        inputs=[video_input, caption_input],
    )

demo.launch(share=True)

