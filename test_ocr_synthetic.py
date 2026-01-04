#!/usr/bin/env python3
"""Test OCR on synthetic PDF files to verify fix."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from extraction.ocr_router import ocr_pdf_multi
from utils.logging import configure_logging, logger

configure_logging()

def test_ocr_on_synthetic():
    """Test OCR on synthetic PDF files."""
    test_pdf = project_root / "data/synthetic/dataset/variation_01/variation_01_original.pdf"
    
    if not test_pdf.exists():
        logger.error(f"Test PDF not found: {test_pdf}")
        return False
    
    logger.info(f"Testing OCR on: {test_pdf}")
    
    try:
        pages = ocr_pdf_multi(test_pdf)
        logger.info(f"Got {len(pages)} pages")
        
        if not pages:
            logger.error("No pages returned from OCR!")
            return False
        
        total_blocks = sum(len(p.blocks) for p in pages)
        total_chars = sum(len(b.text) for p in pages for b in p.blocks)
        
        logger.info(f"Total blocks: {total_blocks}, total chars: {total_chars}")
        
        for i, page in enumerate(pages, 1):
            logger.info(f"Page {i}: {len(page.blocks)} blocks")
            if page.metadata.get('ocr_fallback_reason'):
                logger.warning(f"  Fallback reason: {page.metadata['ocr_fallback_reason']}")
            logger.info(f"  Engine: {page.metadata.get('ocr_engine_used', 'unknown')}")
            logger.info(f"  Method: {page.metadata.get('extraction_method', 'unknown')}")
            
            # Show first few blocks
            for j, block in enumerate(page.blocks[:3], 1):
                text_preview = block.text[:50] + "..." if len(block.text) > 50 else block.text
                logger.info(f"    Block {j}: {text_preview}")
        
        if total_blocks == 0:
            logger.error("OCR failed: no blocks extracted!")
            return False
        
        if total_chars < 100:
            logger.warning(f"OCR extracted only {total_chars} chars (expected more)")
            return False
        
        logger.info("✅ OCR test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"OCR test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_ocr_on_synthetic()
    sys.exit(0 if success else 1)

