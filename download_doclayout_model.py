#!/usr/bin/env python3
"""Download DocLayout-YOLO model from HuggingFace."""
from huggingface_hub import hf_hub_download
from pathlib import Path

print("Downloading DocLayout-YOLO model from HuggingFace...")
print("Repository: juliozhao/DocLayout-YOLO-DocStructBench")

# Download the model
model_path = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt",
    local_dir="./models",
    local_dir_use_symlinks=False
)

print(f"\n✅ Model downloaded successfully!")
print(f"📁 Location: {model_path}")

# Verify the file exists and get size
model_file = Path(model_path)
if model_file.exists():
    size_mb = model_file.stat().st_size / (1024 * 1024)
    print(f"📊 Size: {size_mb:.2f} MB")
else:
    print("❌ Error: Model file not found after download!")
