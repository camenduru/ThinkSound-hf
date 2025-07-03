import argparse
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm  
import logging  
from data_utils.v2a_utils.vggsound_224_no_audio import VGGSound
from data_utils.v2a_utils.feature_utils_224 import FeaturesUtils
import torchaudio
from einops import rearrange
from torch.utils.data.dataloader import default_collate
import numpy as np
from huggingface_hub import hf_hub_download
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def error_avoidance_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)

def main(args):

    print(f"Using root: {args.root}, tsv_path: {args.tsv_path}, save_dir: {args.save_dir}")
    dataset = VGGSound(
        root=args.root,
        tsv_path=args.tsv_path,
        sample_rate=args.sample_rate,
        duration_sec=args.duration_sec,
        audio_samples=args.audio_samples,
        start_row=args.start_row,
        end_row=args.end_row,
        save_dir=args.save_dir
    )
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    dataloader = DataLoader(dataset, batch_size=2, num_workers=8, drop_last=False,collate_fn=error_avoidance_collate)

    print(f"Dataset length: {len(dataset)}")
    feature_extractor = FeaturesUtils(
        vae_ckpt=None,
        vae_config=args.vae_config,
        enable_conditions=True,
        synchformer_ckpt=args.synchformer_ckpt
    ).eval().cuda()

    feature_extractor = feature_extractor

    for i, data in enumerate(tqdm(dataloader, desc="Processing", unit="batch")):
        ids = data['id']  
        with torch.no_grad():
            # audio = data['audio'].cuda(rank, non_blocking=True)
            output = {
                'caption': str(data['caption']),
                'caption_cot': str(data['caption_cot'])
            }
            print(output)

            # latent = feature_extractor.module.encode_audio(audio)
            # output['latent'] = latent.detach().cpu()

            clip_video = data['clip_video'].cuda()
            clip_features = feature_extractor.encode_video_with_clip(clip_video)
            output['metaclip_features'] = clip_features.detach().cpu()

            sync_video = data['sync_video'].cuda()
            sync_features = feature_extractor.encode_video_with_sync(sync_video)
            output['sync_features'] = sync_features.detach().cpu()

            caption = data['caption']
            metaclip_global_text_features, metaclip_text_features = feature_extractor.encode_text(caption)
            output['metaclip_global_text_features'] = metaclip_global_text_features.detach().cpu()
            output['metaclip_text_features'] = metaclip_text_features.detach().cpu()

            caption_cot = data['caption_cot']
            t5_features = feature_extractor.encode_t5_text(caption_cot)
            output['t5_features'] = t5_features.detach().cpu()

            for j in range(len(ids)):
                sample_output = {
                    'id': ids[j],
                    'caption': output['caption'][j],
                    'caption_cot': output['caption_cot'][j],
                    # 'latent': output['latent'][j],
                    'metaclip_features': output['metaclip_features'][j],
                    'sync_features': output['sync_features'][j],
                    'metaclip_global_text_features': output['metaclip_global_text_features'][j],
                    'metaclip_text_features': output['metaclip_text_features'][j],
                    't5_features': output['t5_features'][j],
                }
                # torch.save(sample_output, f'{save_dir}/{ids[j]}.pth')
                np.savez(f'{save_dir}/demo.npz', **sample_output)

        ## test the sync between videos and audios
        # torchaudio.save(f'input_{i}.wav',data['audio'],sample_rate=44100)
        # recon_audio = feature_extractor.decode_audio(latent)
        # recon_audio = rearrange(recon_audio, "b d n -> d (b n)")
        # id = data['id']
        # torchaudio.save(f'recon_{i}.wav',recon_audio.cpu(),sample_rate=44100)
        # os.system(f'ffmpeg -y -i dataset/vggsound/video/train/{id}.mp4 -i recon_{i}.wav -t 9 -map 0:v -map 1:a -c:v copy -c:a aac -strict experimental -shortest out_{i}.mp4')
        # os.system(f'ffmpeg -y -i dataset/vggsound/video/train/{id}.mp4 -i input_{i}.wav -t 9 -map 0:v -map 1:a -c:v copy -c:a aac -strict experimental -shortest input_{i}.mp4')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Video Training Latents')
    parser.add_argument('--root', type=str, default='videos', help='Root directory of the video dataset')
    parser.add_argument('--tsv_path', type=str, default='cot_coarse/cot.csv', help='Path to the TSV file')
    parser.add_argument('--save-dir', type=str, default='results', help='Save Directory')
    parser.add_argument('--sample_rate', type=int, default=44100, help='Sample rate of the audio')
    parser.add_argument('--duration_sec', type=float, default=9.0, help='Duration of the audio in seconds')
    parser.add_argument('--vae_ckpt', type=str, default='ckpts/vae.ckpt', help='Path to the VAE checkpoint')
    parser.add_argument('--vae_config', type=str, default='ThinkSound/configs/model_configs/stable_audio_2_0_vae.json', help='Path to the VAE configuration file')
    parser.add_argument('--synchformer_ckpt', type=str, default='ckpts/synchformer_state_dict.pth', help='Path to the Synchformer checkpoint')
    parser.add_argument('--start-row', type=int, default=0, help='start row')
    parser.add_argument('--end-row', type=int, default=None, help='end row')

    args = parser.parse_args()
    args.audio_samples = int(args.sample_rate * args.duration_sec)

    main(args=args)

