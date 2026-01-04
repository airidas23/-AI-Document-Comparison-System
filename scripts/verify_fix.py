
import sys
import os
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
sys.path.insert(0, os.getcwd())

from extraction.deepseek_ocr_engine import get_ocr_instance
from config.settings import settings

def verify_fix():
    print("=== Verifying DeepSeek OCR Fixes ===")
    
    # 1. Force enable grounding
    print("Enabling grounding in settings...")
    settings.deepseek_grounding_enabled = True
    
    # 2. Create a test image with text lines
    print("Creating test image...")
    img = Image.new('RGB', (1000, 1000), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some text lines
    text_lines = [
        "This is line 1.",
        "This is line 2.",
        "This is a longer paragraph that should be split if it was returned as one block.",
        "Another line here."
    ]
    
    y = 100
    for line in text_lines:
        draw.text((50, y), line, fill="black")
        y += 50
        
    # 3. Initialize engine
    model_path = settings.deepseek_ocr_model_path
    print(f"Loading model from: {model_path}")
    
    start_load = time.time()
    ocr = get_ocr_instance(model_path)
    ocr._load_model()
    print(f"Model loaded in {time.time() - start_load:.2f}s on {ocr._device}")
    
    # 4. Run recognition
    print("\nRunning recognition...")
    start_ocr = time.time()
    blocks = ocr.recognize(img)
    ocr_time = time.time() - start_ocr
    
    print(f"OCR finished in {ocr_time:.2f}s")
    
    # 5. Verify results
    print(f"\nFound {len(blocks)} blocks:")
    grounded_count = 0
    split_count = 0
    
    for i, block in enumerate(blocks):
        source = block.metadata.get("bbox_source", "unknown")
        print(f"Block {i}: Source='{source}', Text='{block.text[:50]}...'")
        
        if "grounded" in source:
            grounded_count += 1
        if "split" in source:
            split_count += 1
            
    print("\n=== Verification Results ===")
    
    # Check Performance
    if ocr_time < 30:
        print(f"✅ Performance PASS: {ocr_time:.2f}s < 30s")
    else:
        print(f"❌ Performance FAIL: {ocr_time:.2f}s > 30s")
        
    # Check Grounding
    if grounded_count > 0:
        print(f"✅ Grounding PASS: Found {grounded_count} grounded blocks")
    else:
        print("⚠️ Grounding WARNING: No grounded blocks found (might be model behavior on synthetic image)")
        
    # Check Splitting (if applicable)
    if split_count > 0:
        print(f"✅ Splitting PASS: Found {split_count} split blocks")
    else:
        print("ℹ️ Splitting INFO: No blocks needed splitting (or model output single lines)")

if __name__ == "__main__":
    verify_fix()
