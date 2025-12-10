#!/usr/bin/env python3
"""Download compatible YOLOv8 DocLayNet model from HuggingFace."""
from huggingface_hub import hf_hub_download
from pathlib import Path

print("=" * 70)
print("Downloading compatible YOLOv8 DocLayNet model...")
print("=" * 70)

# Using neuralshift/doc-layout-yolov8n - a nano model that's compatible
model_path = hf_hub_download(
    repo_id="neuralshift/doc-layout-yolov8n",
    filename="best.pt",
    local_dir="./models",
    local_dir_use_symlinks=False
)

print(f"\n✅ Model downloaded successfully!")
print(f"📁 Location: {model_path}")

# Rename to a descriptive name
final_path = Path("models/yolov8n_doclaynet.pt")
Path(model_path).rename(final_path)

print(f"📁 Renamed to: {final_path}")

# Verify the file exists and get size
if final_path.exists():
    size_mb = final_path.stat().st_size / (1024 * 1024)
    print(f"📊 Size: {size_mb:.2f} MB")
else:
    print("❌ Error: Model file not found after download!")

print("\n" + "=" * 70)
print("✅ Download complete!")
print("=" * 70)
