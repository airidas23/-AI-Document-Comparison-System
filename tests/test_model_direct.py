#!/usr/bin/env python3
"""Direct test of DocLayout-YOLO model."""
from ultralytics import YOLO
from pathlib import Path

print("=" * 70)
print("Direct DocLayout-YOLO Model Test")
print("=" * 70)

model_path = Path("models/doclayout_yolo_docstructbench_imgsz1024.pt")

if not model_path.exists():
    print(f"❌ Model not found: {model_path}")
    exit(1)

print(f"\n📁 Loading model from: {model_path}")
print(f"📊 File size: {model_path.stat().st_size / (1024*1024):.2f} MB")

model = YOLO(str(model_path))

print(f"\n✅ Model loaded successfully!")
print(f"🔢 Number of classes: {len(model.names)}")
print(f"\n📋 Detected classes:")
for idx, name in model.names.items():
    print(f"   {idx}: {name}")

print("\n" + "=" * 70)
print("✅ DocLayout-YOLO model verification complete!")
print("=" * 70)
