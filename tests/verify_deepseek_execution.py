
import os
import sys
from pathlib import Path
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docdiff")

# Add project root to path
sys.path.append(str(Path.cwd()))

from extraction.deepseek_ocr_engine import DeepSeekOCR
from config.settings import settings

def test_deepseek_execution():
    print("Testing DeepSeek execution on Mac...")
    
    # Path to test image
    image_path = Path("models/deepseek-ocr/assets/fig1.png")
    if not image_path.exists():
        print(f"Test image not found at {image_path}")
        # Try another one
        image_path = Path("models/deepseek-ocr/assets/show1.jpg")
        if not image_path.exists():
             print("No test images found in models/deepseek-ocr/assets/")
             return

    print(f"Using image: {image_path}")
    
    try:
        # Initialize
        ocr = DeepSeekOCR(settings.deepseek_ocr_model_path)
        
        # Load image
        img = Image.open(image_path)
        
        # Run recognition
        print("Running recognition (this may take a moment)...")
        blocks = ocr.recognize(img)
        
        print(f"Found {len(blocks)} text blocks.")
        if blocks:
            print("First block text:", blocks[0].text[:50] + "...")
            print("SUCCESS: DeepSeek executed and returned results.")
        else:
            print("WARNING: DeepSeek executed but returned 0 blocks.")
            
    except Exception as e:
        print(f"FAILURE: DeepSeek execution failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_deepseek_execution()
