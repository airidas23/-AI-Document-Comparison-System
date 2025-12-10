"""Legacy OCR processing module - deprecated. Use ocr_router.ocr_pdf_multi() instead."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import List

from comparison.models import PageData
from utils.logging import logger

# Re-export DeepSeekOCR for backward compatibility
from extraction.deepseek_ocr_engine import DeepSeekOCR, get_ocr_instance  # noqa: F401


def ocr_pdf(path: str | Path, model_path: str = None) -> List[PageData]:
    """
    Legacy OCR function - deprecated.
    
    Use ocr_pdf_multi() from ocr_router instead.
    This function is kept for backward compatibility only.
    
    Args:
        path: PDF file path
        model_path: Model path (ignored, kept for compatibility)
    
    Returns:
        List of PageData objects
    """
    warnings.warn(
        "ocr_pdf() is deprecated. Use ocr_pdf_multi() from ocr_router instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Thin wrapper - redirect to router
    from extraction.ocr_router import ocr_pdf_multi
    return ocr_pdf_multi(path)
