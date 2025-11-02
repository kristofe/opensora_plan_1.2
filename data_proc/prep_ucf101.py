import os, json, subprocess, random
from pathlib import Path

SRC_DIR = Path("./source_data/UCF-101")          # raw dataset
DST_DIR = Path("./data/ucf101_256")  # resized/output
DST_DIR.mkdir(parents=True, exist_ok=True)

manifest = []

def class_to_caption(ucf_label):
    # crude captionizer
    words = ucf_label.replace("_", " ")
    return f"a person {words}"

all_videos = list(SRC_DIR.rglob("*.avi"))  # UCF-101 default is .avi
random.shuffle(all_videos)

for vid in all_videos[:1000]:  # take ~1k
    cls = vid.parent.name      # e.g. "PlayingGuitar"
    out_dir = DST_DIR / cls
    out_dir.mkdir(parents=True, exist_ok=True)

    out_mp4 = out_dir / (vid.stem + ".mp4")

    # 1. scale to 256x256, drop fps to 8
    # 2. limit duration to ~1.5s (12 frames @ 8 fps ~1.5s)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(vid),
        "-vf", "scale=256:256:force_original_aspect_ratio=decrease,pad=256:256:(ow-iw)/2:(oh-ih)/2",
        "-r", "8",
        "-t", "1.5",
        "-an",
        str(out_mp4)
    ]
    subprocess.run(cmd, check=True)

    caption = class_to_caption(cls)
    manifest.append({
        "video_path": str(out_mp4),
        "text": caption
    })

with open("data/ucf101_index.json", "w") as f:
    json.dump(manifest, f, indent=2)
