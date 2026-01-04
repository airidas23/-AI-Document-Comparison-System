#!/usr/bin/env python3
"""Test OCR bounding box visualization with the fix applied."""
import sys
import os
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.mark.integration
@pytest.mark.slow
def test_ocr_bbox_viz():
    """Test OCR bounding box visualization after the scaling fix."""
    if os.environ.get("RUN_OCR_BBOX_VIZ") != "1":
        pytest.skip("Set RUN_OCR_BBOX_VIZ=1 to run this integration test")
    from extraction.ocr_visualizer import visualize_ocr_on_pdf_page
    
    # Test PDF (use one from synthetic dataset if available)
    pdf_path = "data/synthetic/dataset/variation_01/variation_01_original.pdf"
    
    if not Path(pdf_path).exists():
        print(f"❌ Test PDF not found: {pdf_path}")
        print("Please provide a valid PDF path to test.")
        pytest.skip("Test PDF not available")
    
    print("=" * 70)
    print("OCR BOUNDING BOX VISUALIZATION TEST")
    print("=" * 70)
    
    try:
        # Test with Paddle OCR (faster than DeepSeek)
        print("\n📄 Processing PDF with Paddle OCR...")
        print(f"   File: {pdf_path}")
        
        output_path = "ocr_viz_test_output.png"
        
        result_img = visualize_ocr_on_pdf_page(
            pdf_path=pdf_path,
            page_num=0,
            ocr_engine="paddle",
            output_path=output_path,
            show_text=False,  # Don't show text labels to keep boxes clean
            show_confidence=False
        )
        
        print(f"\n✅ Visualization complete!")
        print(f"   Output saved to: {output_path}")
        print(f"   Image size: {result_img.width} x {result_img.height} pixels")
        
        print("\n" + "=" * 70)
        print("WHAT TO CHECK:")
        print("=" * 70)
        print("1. Open the generated image: ocr_viz_test_output.png")
        print("2. Bounding boxes should be GREEN rectangles")
        print("3. Boxes should be SMALL, around individual text blocks")
        print("4. Boxes should NOT cover the entire page")
        print("\nIf boxes still cover the entire page, the fix didn't work.")
        print("=" * 70)
        
        assert result_img is not None
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"OCR bbox viz failed: {e}")

if __name__ == "__main__":
    success = test_ocr_bbox_viz()
    sys.exit(0 if success else 1)
