'''
1. Phase 1 – Baseline Teacher Run (what to actually save)

Your Phase 1 bullet list is dead on. Let’s make it concrete.

You’re doing two things with the teacher (Open-Sora Plan v1.2.0):

1.1 Teacher as a VAE/DiT encoder on real UCF-101 clips

→ “student should imitate teacher’s internal representation of real video”
→ this is your DMD-style distillation target.

For each preprocessed clip in ucf101_index.json:

Load the video (12 frames @ 256²).

Run Open-Sora’s video tokenizer / VAE encoder to get latent sequence z_teacher.

Save that latent tensor to disk next to the clip.

(Optionally) run 1–2 denoising steps of the teacher diffusion on that clip in “reconstruction” mode to also cache intermediate noise states. That becomes a supervision signal for the student’s denoiser head.
'''

import torch, json, os
from opensora.models import TeacherModel   # placeholder import
from opensora.utils.video import load_video_tensor # e.g. [T,C,H,W] bf16

teacher = TeacherModel.from_pretrained("opensora_plan_v1_2_0.pt").to("cuda").eval()
torch_dtype = torch.bfloat16

with open("data/ucf101_index.json") as f:
    manifest = json.load(f)

os.makedirs("data/teacher_latents", exist_ok=True)

for item in manifest:
    vid_path = item["video_path"]
    txt      = item["text"]

    # 1. load frames -> tensor [T,C,H,W], T≈12, C=3, H=W=256, bf16
    video_tensor = load_video_tensor(vid_path, dtype=torch_dtype).to("cuda")

    # 2. encode frames to latent tokens
    with torch.no_grad():
        z_teacher = teacher.encode_video(video_tensor)       # [T, latent_dim, h', w']
        txt_emb   = teacher.encode_text(txt)                 # [text_dim]

        # optional: run partial diffusion steps to capture intermediate noise / eps predictions
        # eps_pred_seq = teacher.forward_denoise_steps(video_tensor, txt_emb, steps=[980, 960])

    torch.save(
        {
            "z_teacher": z_teacher.cpu(),
            "text_emb": txt_emb.cpu(),
            # "eps_seq": eps_pred_seq.cpu(),  # if you captured it
        },
        f"data/teacher_latents/{os.path.basename(vid_path)}.pt"
    )


