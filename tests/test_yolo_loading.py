#!/usr/bin/env python3
"""Test script to verify YOLO model loading."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from extraction.layout_analyzer import load_yolo_model, analyze_layout
from utils.logging import logger

def test_yolo_model_loading():
    """Test if YOLO model loads correctly."""
    logger.info("=" * 60)
    logger.info("Testing YOLO Model Loading")
    logger.info("=" * 60)
    
    # Test 1: Load the model
    logger.info("Test 1: Loading YOLO model...")
    model = load_yolo_model()
    
    if model is not None:
        logger.info("✅ SUCCESS: YOLO model loaded successfully!")
        logger.info(f"Model type: {type(model)}")
    else:
        logger.error("❌ FAILED: YOLO model failed to load")
        return False
    
    # Test 2: Try to analyze a PDF
    logger.info("\nTest 2: Analyzing a PDF with YOLO...")
    test_pdf = project_root / "AI Document Comparison System Prototype Plan.pdf"
    
    if not test_pdf.exists():
        logger.warning(f"Test PDF not found: {test_pdf}")
        logger.info("Skipping PDF analysis test")
        return True
    
    try:
        pages = analyze_layout(test_pdf, use_layoutparser=True)
        logger.info(f"✅ SUCCESS: Analyzed {len(pages)} pages")
        
        # Check if YOLO was used
        if pages:
            method = pages[0].metadata.get("layout_method", "unknown")
            logger.info(f"Layout method used: {method}")
            
            if method == "yolo":
                logger.info("✅ YOLO was successfully used for layout analysis!")
            else:
                logger.warning(f"⚠️  Expected 'yolo' but got '{method}'")
        
        return True
    except Exception as e:
        logger.error(f"❌ FAILED: Error analyzing PDF: {e}")
        return False

if __name__ == "__main__":
    success = test_yolo_model_loading()
    sys.exit(0 if success else 1)
