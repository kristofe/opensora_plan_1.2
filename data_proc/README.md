# Teacher Data Generation

Scripts for creating teacher training data for video diffusion distillation.

## Two Approaches

### 1. Encode Real Videos (Phase 1.1)

Encode UCF-101 videos into teacher's latent space:

```bash
python data_proc/create_teacher_data.py \
    --manifest_path data/ucf101_index.json \
    --vae_path models/vae \
    --text_encoder_name_1 google/mt5-xxl \
    --output_dir data/teacher_latents \
    --device cuda \
    --dtype bfloat16
```

**Output:** `data/teacher_latents/*.pt` files with:
- `z_teacher`: VAE latents
- `text_emb`: Text embeddings
- `text_attention_mask`: Attention mask

### 2. Generate Synthetic Videos (Phase 1.2)

Generate new videos from text with teacher model:

```bash
python data_proc/generate_teacher_videos.py \
    --manifest_path data/ucf101_index.json \
    --vae_path models/vae \
    --dit_model_path models/diffusion \
    --text_encoder_name_1 google/mt5-xxl \
    --output_dir data/teacher_synthetic \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --save_frames \
    --device cuda
```

**Output:** `data/teacher_synthetic/{id}/` directories with:
- `generation_data.pt`: Video, latents, trajectory, embeddings
- `frames/`: Individual PNG frames (if `--save_frames`)

## Why Both?

| Type | Benefits |
|------|----------|
| **Real** | Actual data, real-world diversity, ground truth |
| **Synthetic** | Perfect alignment, no copyright, includes diffusion trajectory |

**Recommendation:** Use both for complementary training signals.

## Manifest Format

```json
[
  {
    "id": "basketball_001",
    "text": "A person playing basketball",
    "video_path": "data/videos/basketball.avi"  // Required for encoding
  }
]
```

## Utilities

### Visualize Generated Data

```bash
python data_proc/visualize_teacher_data.py \
    --data_path data/teacher_synthetic/item_00000/generation_data.pt \
    --output_dir visualizations/ \
    --show_latents \
    --show_text
```

### Load Data for Training

```python
from torch.utils.data import DataLoader
from data_proc.teacher_dataset import TeacherDataset, collate_teacher_data

dataset = TeacherDataset(
    real_data_dir="data/teacher_latents",
    synthetic_data_dir="data/teacher_synthetic",
    mode="both"
)

loader = DataLoader(dataset, batch_size=8, collate_fn=collate_teacher_data)

for batch in loader:
    if 'real' in batch:
        z_teacher = batch['real']['z_teacher']
        # Train on real data
    
    if 'synthetic' in batch:
        video = batch['synthetic']['video']
        trajectory = batch['synthetic']['trajectory']
        # Train on synthetic + trajectory
```

## Performance Tips

**Memory:** Use `--enable_tiling` and reduce resolution/frames  
**Speed:** Use DDIM scheduler, fewer inference steps  
**Quality:** Increase inference steps, adjust guidance scale

## Files

- `create_teacher_data.py` - Encode real videos
- `generate_teacher_videos.py` - Generate synthetic videos  
- `teacher_dataset.py` - Dataset loader for training
- `visualize_teacher_data.py` - Visualization tool

---

**Questions?** Start with the example commands above, then customize as needed.

