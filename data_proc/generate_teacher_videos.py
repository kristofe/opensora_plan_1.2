'''
Generate Synthetic Videos with Teacher Model

Phase 1.2: Use teacher model to generate synthetic training data from text prompts.

Usage:
python data_proc/generate_teacher_videos.py \
    --manifest_path data/ucf101_index.json \
    --vae_path models/vae \
    --dit_model_path models/diffusion \
    --text_encoder_name_1 google/mt5-xxl \
    --output_dir data/teacher_synthetic \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --save_frames \
    --device cuda \
    --dtype bfloat16
'''

import argparse
import json
import os
import torch
from torchvision.utils import save_image

from opensora.sample.pipeline_opensora import OpenSoraPipeline

# Import shared functions from create_teacher_data
import sys
sys.path.insert(0, os.path.dirname(__file__))
from create_teacher_data import encode_text


def save_video_frames(video_tensor: torch.Tensor, output_dir: str, prefix: str = "frame"):
    """Save video frames as individual PNG images."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy if it's a tensor
    if isinstance(video_tensor, torch.Tensor):
        video_tensor = video_tensor.cpu()
    
    # Handle different tensor shapes
    # Pipeline returns (T, H, W, C) format from decode
    if video_tensor.ndim == 4:
        # Check if it's (T, H, W, C) or (T, C, H, W) or (C, T, H, W) or (B, C, H, W)
        if video_tensor.shape[-1] == 3 or video_tensor.shape[-1] == 4:
            # (T, H, W, C) format - convert to (T, C, H, W)
            video_tensor = video_tensor.permute(0, 3, 1, 2)
        elif video_tensor.shape[1] == 3 or video_tensor.shape[1] == 4:
            # Already (T, C, H, W) or (B, C, H, W)
            pass
        else:
            # (C, T, H, W) - transpose to (T, C, H, W)
            video_tensor = video_tensor.permute(1, 0, 2, 3)
    elif video_tensor.ndim == 5:
        # (B, T, H, W, C) or (B, T, C, H, W) or (B, C, T, H, W)
        video_tensor = video_tensor.squeeze(0)  # Remove batch
        if video_tensor.shape[-1] == 3 or video_tensor.shape[-1] == 4:
            # (T, H, W, C) format - convert to (T, C, H, W)
            video_tensor = video_tensor.permute(0, 3, 1, 2)
        elif video_tensor.shape[1] == 3 or video_tensor.shape[1] == 4:
            # Already (T, C, H, W)
            pass
        else:
            # (C, T, H, W) - transpose to (T, C, H, W)
            video_tensor = video_tensor.permute(1, 0, 2, 3)
    
    # Denormalize from [-1, 1] to [0, 1]
    video_tensor = (video_tensor + 1.0) / 2.0
    video_tensor = torch.clamp(video_tensor, 0, 1)
    
    # video_tensor should now be (T, C, H, W)
    for t in range(video_tensor.shape[0]):
        frame_path = os.path.join(output_dir, f"{prefix}_{t:04d}.png")
        save_image(video_tensor[t], frame_path)


def generate_video_from_text(
    text, pipeline, num_frames=12, height=256, width=256,
    num_inference_steps=50, guidance_scale=7.5, device="cuda"
):
    """Generate video from text using teacher pipeline."""
    
    # Use the pipeline to generate video
    result = pipeline(
        prompt=text,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_type="pt",  # Return pytorch tensors
        device=device,
    )
    
    # Extract the video
    generated_video = result.videos[0]  # First (and only) sample
    
    # Get text embeddings for saving
    text_encoder_1 = pipeline.text_encoder
    tokenizer_1 = pipeline.tokenizer
    text_encoder_2 = pipeline.text_encoder_2 if hasattr(pipeline, 'text_encoder_2') else None
    tokenizer_2 = pipeline.tokenizer_2 if hasattr(pipeline, 'tokenizer_2') else None
    
    text_results = encode_text(
        text, text_encoder_1, tokenizer_1, text_encoder_2, tokenizer_2, 
        device, pipeline.transformer.dtype
    )
    
    return {
        'generated_video': generated_video.cpu(),
        'text_embeddings': text_results
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic videos with teacher model")
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vae_path", type=str, default="models/vae")
    parser.add_argument("--vae_model", type=str, default="CausalVAEModel_D4_2x8x8")
    parser.add_argument("--dit_model_path", type=str, default="models/diffusion")
    parser.add_argument("--text_encoder_name_1", type=str, default="google/mt5-xxl")
    parser.add_argument("--text_encoder_name_2", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_frames", type=int, default=13, help="Number of frames (num_frames-1 must be divisible by 4)")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--save_frames", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()
    
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)
    
    # Load components manually
    print("Loading VAE...")
    from opensora.models.causalvideovae import ae_wrapper, ae_stride_config
    vae = ae_wrapper[args.vae_model](args.vae_path, cache_dir=args.cache_dir)
    vae.vae = vae.vae.to(device, dtype=torch_dtype).eval()
    # Add vae_scale_factor attribute needed by pipeline (to both wrapper and inner model)
    vae.vae_scale_factor = ae_stride_config[args.vae_model]
    vae.vae.vae_scale_factor = ae_stride_config[args.vae_model]
    
    print(f"Loading text encoder: {args.text_encoder_name_1}")
    if 'mt5' in args.text_encoder_name_1.lower():
        from transformers import MT5EncoderModel, MT5Tokenizer
        text_encoder_1 = MT5EncoderModel.from_pretrained(
            args.text_encoder_name_1, cache_dir=args.cache_dir, torch_dtype=torch_dtype
        ).eval().to(device)
        tokenizer_1 = MT5Tokenizer.from_pretrained(args.text_encoder_name_1, cache_dir=args.cache_dir)
    else:
        from transformers import T5EncoderModel, AutoTokenizer
        text_encoder_1 = T5EncoderModel.from_pretrained(
            args.text_encoder_name_1, cache_dir=args.cache_dir, torch_dtype=torch_dtype
        ).eval().to(device)
        tokenizer_1 = AutoTokenizer.from_pretrained(args.text_encoder_name_1, cache_dir=args.cache_dir)
    
    text_encoder_2, tokenizer_2 = None, None
    if args.text_encoder_name_2:
        print(f"Loading text encoder 2: {args.text_encoder_name_2}")
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            args.text_encoder_name_2, cache_dir=args.cache_dir, torch_dtype=torch_dtype
        ).eval().to(device)
        tokenizer_2 = CLIPTokenizer.from_pretrained(args.text_encoder_name_2, cache_dir=args.cache_dir)
    
    print(f"Loading diffusion model from {args.dit_model_path}")
    from opensora.models.diffusion import Diffusion_models_class
    config_path = os.path.join(args.dit_model_path, "config.json")
    model_class_name = "OpenSoraT2V_v1_3"
    
    is_local_path = os.path.exists(args.dit_model_path) and os.path.isdir(args.dit_model_path)
    
    if is_local_path and os.path.exists(config_path):
        with open(config_path) as f:
            config_data = json.load(f)
            model_class_name = config_data.get("_class_name", model_class_name)
            if model_class_name == "OpenSoraT2V":
                model_class_name = "OpenSoraT2V_v1_3"
    
    model_class = Diffusion_models_class.get(model_class_name)
    if not model_class:
        from opensora.models.diffusion.opensora_v1_3.modeling_opensora import OpenSoraT2V_v1_3
        model_class = OpenSoraT2V_v1_3
    
    if is_local_path:
        diffusion_model = model_class.from_pretrained(
            args.dit_model_path, 
            local_files_only=True,
            torch_dtype=torch_dtype
        ).eval().to(device)
    else:
        diffusion_model = model_class.from_pretrained(
            args.dit_model_path, 
            cache_dir=args.cache_dir, 
            torch_dtype=torch_dtype
        ).eval().to(device)
    
    # Monkey-patch: Fix sparse_n KeyError by overriding the forward method
    # The opensora model only prepares masks for sparse_n=[1,4] but uses sparse_n=2 by default
    original_forward = diffusion_model.forward
    def patched_forward(self, *args, **kwargs):
        # Temporarily change sparse_n to 1 for all blocks to avoid KeyError
        original_sparse_n = []
        for block in self.transformer_blocks:
            original_sparse_n.append(block.attn1.processor.sparse_n)
            if block.attn1.processor.sparse_n == 2:
                block.attn1.processor.sparse_n = 1
                block.attn2.processor.sparse_n = 1
        
        # Call original forward
        result = original_forward(*args, **kwargs)
        
        # Restore original sparse_n values
        for block, orig_n in zip(self.transformer_blocks, original_sparse_n):
            block.attn1.processor.sparse_n = orig_n
            block.attn2.processor.sparse_n = orig_n
        
        return result
    
    # Bind the patched method
    import types
    diffusion_model.forward = types.MethodType(patched_forward, diffusion_model)
    
    print("Loading scheduler...")
    from diffusers import DDIMScheduler
    scheduler = DDIMScheduler()
    
    # Build pipeline from components
    print("Building pipeline...")
    pipeline = OpenSoraPipeline(
        vae=vae,  # Pass the wrapper, not vae.vae
        text_encoder=text_encoder_1,
        tokenizer=tokenizer_1,
        transformer=diffusion_model,
        scheduler=scheduler,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
    )
    
    # Load manifest
    with open(args.manifest_path) as f:
        manifest = json.load(f)
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nGenerating {len(manifest)} videos...\n")
    
    # Generate videos
    for idx, item in enumerate(manifest):
        text = item["text"]
        item_id = item.get("id", f"item_{idx:05d}")
        
        print(f"[{idx+1}/{len(manifest)}] '{text}'")
        
        try:
            results = generate_video_from_text(
                text, pipeline,
                args.num_frames, args.height, args.width,
                args.num_inference_steps, args.guidance_scale, device
            )
            
            item_dir = os.path.join(args.output_dir, item_id)
            os.makedirs(item_dir, exist_ok=True)
            
            if args.save_frames:
                frames_dir = os.path.join(item_dir, "frames")
                save_video_frames(results['generated_video'], frames_dir)
            
            save_dict = {
                "text": text,
                "generated_video": results['generated_video'],
                "text_emb": results['text_embeddings']['text_emb'],
                "text_attention_mask": results['text_embeddings']['text_attention_mask'],
            }
            if 'text_emb_2' in results['text_embeddings']:
                save_dict['text_emb_2'] = results['text_embeddings']['text_emb_2']
                save_dict['text_attention_mask_2'] = results['text_embeddings']['text_attention_mask_2']
            
            torch.save(save_dict, os.path.join(item_dir, "generation_data.pt"))
            print(f"  ✓ Saved to {item_dir}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nDone!")


if __name__ == "__main__":
    main()

