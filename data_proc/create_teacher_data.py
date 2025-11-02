'''
Usage Example
python data_proc/create_teacher_data.py \
    --manifest_path data/ucf101_index.json \
    --vae_path /path/to/vae/checkpoint \
    --text_encoder_name_1 google/mt5-xxl \
    --output_dir data/teacher_latents \
    --device cuda \
    --dtype bfloat16

Open-Sora Plan v1.2.0 uses mT5-XXL for multilingual adaptation.
'''
'''
1. Phase 1 – Baseline Teacher Run (what to actually save)

Your Phase 1 bullet list is dead on. Let's make it concrete.

You're doing two things with the teacher (Open-Sora Plan v1.2.0):

1.1 Teacher as a VAE/DiT encoder on real UCF-101 clips

→ "student should imitate teacher's internal representation of real video"
→ this is your DMD-style distillation target.

For each preprocessed clip in ucf101_index.json:

Load the video (12 frames @ 256²).

Run Open-Sora's video tokenizer / VAE encoder to get latent sequence z_teacher.

Save that latent tensor to disk next to the clip.

(Optionally) run 1–2 denoising steps of the teacher diffusion on that clip in "reconstruction" mode to also cache intermediate noise states. That becomes a supervision signal for the student's denoiser head.
'''

import argparse
import json
import os

import numpy as np
import torch
from decord import VideoReader, cpu
from torchvision.transforms import Compose, Lambda
from transformers import T5EncoderModel, CLIPTextModelWithProjection, CLIPTokenizer

# Fix for huggingface_hub compatibility - cached_download was removed in v0.26.0
import huggingface_hub
if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

from opensora.dataset.transform import ToTensorVideo, CenterCropResizeVideo
from opensora.models.causalvideovae import ae_wrapper


def read_video(video_path: str, num_frames: int = 12, sample_rate: int = 1) -> torch.Tensor:
    """Load video frames using decord.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        sample_rate: Frame interval
    
    Returns:
        Tensor of shape (C, T, H, W) where C=3, T=num_frames
    """
    decord_vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(decord_vr)
    sample_frames_len = sample_rate * num_frames

    if total_frames >= sample_frames_len:
        s = 0
        e = sample_frames_len
    else:
        s = 0
        e = total_frames
        num_frames = int(total_frames / sample_rate)
        print(f"Warning: Video {video_path} has only {total_frames} frames, using {num_frames} frames")

    frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)
    video_data = decord_vr.get_batch(frame_id_list).asnumpy()
    video_data = torch.from_numpy(video_data)
    video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
    return video_data


def preprocess_video(video_data: torch.Tensor, height: int = 256, width: int = 256) -> torch.Tensor:
    """Preprocess video to format expected by VAE.
    
    Args:
        video_data: Tensor of shape (C, T, H, W)
        height: Target height
        width: Target width
    
    Returns:
        Tensor of shape (B, C, T, H, W) where B=1, normalized to [-1, 1]
    """
    # Transform expects (T, C, H, W) format, so transpose from (C, T, H, W)
    video_data = video_data.permute(1, 0, 2, 3)  # (C, T, H, W) -> (T, C, H, W)
    
    transform = Compose([
        ToTensorVideo(),  # Converts uint8 [0, 255] -> float [0, 1]
        CenterCropResizeVideo((height, width)),
        Lambda(lambda x: 2. * x - 1.)  # Normalize [0, 1] -> [-1, 1]
    ])
    video_outputs = transform(video_data)
    
    # Transform returns (T, C, H, W), convert back to (C, T, H, W) then add batch dim
    video_outputs = video_outputs.permute(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)
    video_outputs = torch.unsqueeze(video_outputs, 0)  # Add batch dimension -> (B, C, T, H, W)
    return video_outputs


def encode_text(text: str, text_encoder, tokenizer, text_encoder_2=None, tokenizer_2=None, device="cuda", dtype=torch.bfloat16):
    """Encode text using T5/mT5-xxl and optionally CLIP text encoders.
    
    Args:
        text: Input text string
        text_encoder: T5/mT5-xxl text encoder
        tokenizer: T5/mT5-xxl tokenizer
        text_encoder_2: Optional CLIP text encoder
        tokenizer_2: Optional CLIP tokenizer
        device: Device to run on
        dtype: Data type
    
    Returns:
        Dictionary with 'text_emb' (T5/mT5-xxl embeddings) and optionally 'text_emb_2' (CLIP embeddings)
    """
    results = {}
    
    # Encode with T5/mT5-xxl
    text_inputs = tokenizer(
        text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    
    with torch.no_grad():
        text_emb = text_encoder(text_input_ids, attention_mask=attention_mask)[0]
        text_emb = text_emb.to(dtype=dtype)
        results['text_emb'] = text_emb.cpu()
        results['text_attention_mask'] = attention_mask.cpu()
    
    # Encode with CLIP if available
    if text_encoder_2 is not None and tokenizer_2 is not None:
        text_inputs_2 = tokenizer_2(
            text,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs_2.input_ids.to(device)
        attention_mask_2 = text_inputs_2.attention_mask.to(device)
        
        with torch.no_grad():
            text_emb_2 = text_encoder_2(text_input_ids_2, attention_mask=attention_mask_2)[0]
            text_emb_2 = text_emb_2.unsqueeze(1).to(dtype=dtype)  # (B, D) -> (B, 1, D) for CLIP
            results['text_emb_2'] = text_emb_2.cpu()
            results['text_attention_mask_2'] = attention_mask_2.cpu()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Create teacher data by encoding UCF-101 videos")
    parser.add_argument("--manifest_path", type=str, default="data/ucf101_index.json",
                        help="Path to manifest JSON file")
    parser.add_argument("--output_dir", type=str, default="data/teacher_latents",
                        help="Directory to save encoded latents")
    parser.add_argument("--vae_model", type=str, default="CausalVAEModel_D4_2x8x8",
                        help="VAE model type")
    parser.add_argument("--vae_path", type=str, default="models/vae",
                        help="Path to VAE model checkpoint")
    parser.add_argument("--text_encoder_name_1", type=str, default="google/mt5-xxl",
                        help="T5/mT5 text encoder name (use mT5-XXL for multilingual adaptation)")
    parser.add_argument("--text_encoder_name_2", type=str, default=None,
                        help="Optional CLIP text encoder name")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for models")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"],
                        help="Data type for computation")
    parser.add_argument("--num_frames", type=int, default=12,
                        help="Number of frames to extract from video")
    parser.add_argument("--sample_rate", type=int, default=1,
                        help="Frame sampling rate")
    parser.add_argument("--height", type=int, default=256,
                        help="Video height")
    parser.add_argument("--width", type=int, default=256,
                        help="Video width")
    parser.add_argument("--enable_tiling", action="store_true",
                        help="Enable VAE tiling for memory efficiency")
    
    args = parser.parse_args()
    
    # Setup dtype
    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    device = torch.device(args.device)
    
    # Load VAE
    print(f"Loading VAE model: {args.vae_model} from {args.vae_path}")
    vae = ae_wrapper[args.vae_model](args.vae_path, cache_dir=args.cache_dir)
    vae.vae = vae.vae.to(device=device, dtype=torch_dtype).eval()
    if args.enable_tiling:
        vae.vae.enable_tiling()
        print("VAE tiling enabled")
    
    # Load text encoders
    print(f"Loading text encoder 1: {args.text_encoder_name_1}")
    if 'mt5' in args.text_encoder_name_1.lower():
        from transformers import MT5EncoderModel, MT5Tokenizer as MT5TokenizerClass
        text_encoder_1 = MT5EncoderModel.from_pretrained(
            args.text_encoder_name_1, cache_dir=args.cache_dir, torch_dtype=torch_dtype
        ).eval().to(device)
        tokenizer_1 = MT5TokenizerClass.from_pretrained(
            args.text_encoder_name_1, cache_dir=args.cache_dir
        )
    else:
        from transformers import T5EncoderModel, AutoTokenizer
        text_encoder_1 = T5EncoderModel.from_pretrained(
            args.text_encoder_name_1, cache_dir=args.cache_dir, torch_dtype=torch_dtype
        ).eval().to(device)
        tokenizer_1 = AutoTokenizer.from_pretrained(
            args.text_encoder_name_1, cache_dir=args.cache_dir
        )
    
    text_encoder_2 = None
    tokenizer_2 = None
    if args.text_encoder_name_2 is not None:
        print(f"Loading text encoder 2: {args.text_encoder_name_2}")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            args.text_encoder_name_2, cache_dir=args.cache_dir, torch_dtype=torch_dtype
        ).eval().to(device)
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            args.text_encoder_name_2, cache_dir=args.cache_dir
        )
    
    # Load manifest
    with open(args.manifest_path) as f:
        manifest = json.load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Processing {len(manifest)} videos...")
    
    # Process each video
    for idx, item in enumerate(manifest):
        vid_path = item["video_path"]
        txt = item["text"]
        
        print(f"[{idx+1}/{len(manifest)}] Processing {vid_path}")
        
        try:
            # 1. Load and preprocess video
            video_data = read_video(vid_path, num_frames=args.num_frames, sample_rate=args.sample_rate)
            video_tensor = preprocess_video(video_data, height=args.height, width=args.width)
            video_tensor = video_tensor.to(device, dtype=torch_dtype)  # (B, C, T, H, W)
            
            # 2. Encode video to latents
            with torch.no_grad():
                z_teacher = vae.encode(video_tensor)  # Returns latents tensor
            
            # 3. Encode text
            text_results = encode_text(
                txt, text_encoder_1, tokenizer_1, 
                text_encoder_2, tokenizer_2,
                device=device, dtype=torch_dtype
            )
            
            # 4. Save results
            output_path = os.path.join(args.output_dir, f"{os.path.basename(vid_path)}.pt")
            save_dict = {
                "z_teacher": z_teacher.cpu(),
                "text_emb": text_results['text_emb'],
                "text_attention_mask": text_results['text_attention_mask'],
            }
            if 'text_emb_2' in text_results:
                save_dict['text_emb_2'] = text_results['text_emb_2']
                save_dict['text_attention_mask_2'] = text_results['text_attention_mask_2']
            
            torch.save(save_dict, output_path)
            print(f"  Saved to {output_path}")
            
        except Exception as e:
            print(f"  Error processing {vid_path}: {e}")
            continue
    
    print("Done!")


if __name__ == "__main__":
    main()


'''
NEXT STEPS:
1.2 Teacher as a generator, driven by captions (text→video)

This is slightly different and also valuable:

For each text in your manifest:

Ask teacher to generate a clip (text→video, standard T2V sampling).

Save both:

the generated RGB frames (teacher’s “clean output”),

and the diffusion latents + text embeddings used during sampling.

Why?
Because now you have perfectly aligned (prompt, generated_video, z_teacher_generated).
This gives you:

clean, consistent synthetic data in the teacher’s own style,

no copyright headache,

and great input for Self-Forcing later (see below).

So Phase 1 produces two parallel corpora:

Real UCF-101 clips encoded into latents.

Synthetic teacher generations from the same action captions.

That’s gold. Keep both.

'''