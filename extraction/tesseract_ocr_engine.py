"""Tesseract OCR engine for OCR processing with bbox support."""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict

from comparison.models import PageData, TextBlock
from config.settings import settings
from utils.logging import logger


def ocr_pdf(path: str | Path) -> List[PageData]:
    """
    Process a PDF through Tesseract OCR and return PageData.
    
    Args:
        path: Path to PDF file
    
    Returns:
        List of PageData objects with extracted text blocks
    """
    path = Path(path)
    logger.info("Running Tesseract OCR on PDF: %s", path)
    
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError as exc:
        raise RuntimeError(
            "pytesseract is required. Install via `pip install pytesseract`. "
            "Also ensure Tesseract binary is installed (brew install tesseract on Mac)."
        ) from exc
    
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for OCR rendering. Install via `pip install PyMuPDF`."
        ) from exc
    
    doc = fitz.open(path)
    pages: List[PageData] = []
    
    for page in doc:
        # Render page at moderate resolution for OCR (150 DPI is sufficient)
        # Higher DPI = much slower processing
        pix = page.get_pixmap(dpi=150)
        
        # Convert pixmap to PIL Image for Tesseract
        from PIL import Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Run OCR with bounding box data
        ocr_data = pytesseract.image_to_data(
            img,
            lang=settings.tesseract_lang,
            output_type=pytesseract.Output.DICT
        )
        
        # Convert Tesseract results to TextBlocks
        text_blocks = _tesseract_data_to_text_blocks(ocr_data, pix.width, pix.height)
        
        page_data = PageData(
            page_num=page.number + 1,
            width=page.rect.width,
            height=page.rect.height,
            blocks=text_blocks,
        )
        page_data.metadata = {
            "extraction_method": "ocr_tesseract",
            "ocr_engine_used": "tesseract",
            "dpi": 150,
        }
        pages.append(page_data)
    
    doc.close()
    logger.info("Tesseract OCR processed %d pages", len(pages))
    return pages


def _tesseract_data_to_text_blocks(
    ocr_data: dict,
    img_width: float,
    img_height: float
) -> List[TextBlock]:
    """
    Convert Tesseract OCR data to TextBlock format.
    
    Tesseract returns dict with keys: 'left', 'top', 'width', 'height', 'text', 'conf', etc.
    
    We convert to our format: {"x": x, "y": y, "width": w, "height": h}
    """
    text_blocks = []
    
    n_boxes = len(ocr_data['text'])
    
    scale_factor = 72.0 / 150.0  # Convert 150 DPI pixels to 72 DPI points

    for i in range(n_boxes):
        text = ocr_data['text'][i].strip()
        conf = int(ocr_data['conf'][i]) if ocr_data['conf'][i] != '-1' else 0
        
        # Skip empty text or low confidence
        if not text or conf < 30:  # Minimum confidence threshold
            continue
        
        # Get bounding box coordinates in pixels (150 DPI)
        left = ocr_data['left'][i]
        top = ocr_data['top'][i]
        width = ocr_data['width'][i]
        height = ocr_data['height'][i]
        
        # Skip very small boxes (likely noise)
        if width < 5 or height < 5:
            continue
        
        # Convert to PDF points (72 DPI)
        bbox = {
            "x": float(left) * scale_factor,
            "y": float(top) * scale_factor,
            "width": float(width) * scale_factor,
            "height": float(height) * scale_factor,
        }
        
        # Create TextBlock with metadata
        block = TextBlock(
            text=text,
            bbox=bbox,
            style=None,
            metadata={
                "ocr_engine": "tesseract",
                "bbox_source": "exact",
                "confidence": conf,
            }
        )
        text_blocks.append(block)
    
    return text_blocks
