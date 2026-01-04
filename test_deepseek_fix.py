#!/usr/bin/env python3
"""Test DeepSeek-OCR specifically to verify CUDA/MPS fix."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from extraction.ocr_router import ocr_pdf_multi, is_deepseek_available
from utils.logging import configure_logging, logger

configure_logging()

def test_deepseek_fix(monkeypatch):
    """Regression: DeepSeek timeout should not end as 'all engines failed' in AUTO mode.

    On Apple Silicon, DeepSeek may be slow. This test intentionally sets a very
    small DeepSeek timeout and asserts we still get results via deterministic
    fallback (typically PaddleOCR).
    """
    import pytest
    from config.settings import settings

    test_pdf = project_root / "data/synthetic/dataset/variation_01/variation_01_original.pdf"
    
    if not test_pdf.exists():
        pytest.skip(f"Test PDF not found: {test_pdf}")
    
    logger.info(f"Testing OCR routing (DeepSeek-first + fallback) on: {test_pdf}")
    
    try:
        # Force policy to allow fallback and keep the DeepSeek attempt bounded.
        monkeypatch.setattr(settings, "ocr_scanned_policy", "auto_fallback", raising=False)
        monkeypatch.setattr(settings, "ocr_scanned_fallback_chain", ["paddle", "tesseract"], raising=False)
        if is_deepseek_available():
            monkeypatch.setattr(settings, "deepseek_timeout_sec_per_page", 1, raising=False)
            monkeypatch.setattr(settings, "deepseek_hard_timeout", True, raising=False)

        pages = ocr_pdf_multi(test_pdf, engine_priority=["deepseek"], prefer_digital=False)
        logger.info(f"Got {len(pages)} pages")
        
        assert pages, "No pages returned from OCR"
        
        total_blocks = sum(len(p.blocks) for p in pages)
        total_chars = sum(len(b.text) for p in pages for b in p.blocks)
        
        logger.info(f"Total blocks: {total_blocks}, total chars: {total_chars}")
        
        for i, page in enumerate(pages, 1):
            logger.info(f"Page {i}: {len(page.blocks)} blocks")
            logger.info(f"  Engine: {page.metadata.get('ocr_engine_used', 'unknown')}")
            logger.info(f"  Policy: {page.metadata.get('ocr_policy', '')}")
            if page.metadata.get("ocr_failure_reason"):
                logger.warning(f"  Failure reason: {page.metadata['ocr_failure_reason']}")
            if page.metadata.get("ocr_attempts"):
                logger.info(f"  Attempts: {len(page.metadata['ocr_attempts'])}")

            # Show first few blocks
            for j, block in enumerate(page.blocks[:3], 1):
                text_preview = block.text[:50] + "..." if len(block.text) > 50 else block.text
                logger.info(f"    Block {j}: {text_preview}")
        
        assert total_blocks > 0, "OCR failed: no blocks extracted"
        # In auto_fallback, DeepSeek-first should still produce a usable result
        # even if DeepSeek is slow/unavailable.
        assert any(
            (p.metadata or {}).get("ocr_engine_used") in {"paddle", "deepseek", "tesseract"}
            for p in pages
        ), "Expected at least one known OCR engine to be recorded"
        
        logger.info("✅ OCR routing test PASSED (DeepSeek-first + fallback)")
        return
        
    except Exception as e:
        logger.error(f"DeepSeek-OCR test FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    success = test_deepseek_fix()
    sys.exit(0 if success else 1)


