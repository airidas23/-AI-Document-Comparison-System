"""PaddleOCR engine for OCR processing with natural bbox support."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from comparison.models import PageData, TextBlock
from config.settings import settings
from utils.logging import logger

# Module-level cache for PaddleOCR instance
_ocr_instance: Optional["PaddleOCR"] = None


def _get_ocr():
    """
    Get or create a cached PaddleOCR instance.
    Caching avoids slow model reloading on every call.
    """
    global _ocr_instance
    
    if _ocr_instance is None:
        import time
        init_start = time.time()
        logger.info("[PaddleOCR] Initializing PaddleOCR (first time, may take 1-2 minutes)...")
        logger.info("[PaddleOCR] Language: %s", settings.paddle_ocr_lang)
        
        logger.debug("[PaddleOCR] Importing paddleocr module...")
        import_start = time.time()
        from paddleocr import PaddleOCR
        logger.debug("[PaddleOCR] Import took %.2fs", time.time() - import_start)
        
        # Initialize PaddleOCR (v3.x API) 
        # Disable optional models for faster initialization
        
        # Suppress PaddleOCR verbose logging
        import logging
        logging.getLogger("ppocr").setLevel(logging.WARNING)
        
        logger.info("[PaddleOCR] Creating PaddleOCR instance...")
        create_start = time.time()
        _ocr_instance = PaddleOCR(
            use_doc_orientation_classify=False,  # Skip document orientation
            use_doc_unwarping=False,             # Skip document unwarping
            use_textline_orientation=False,      # Skip textline orientation
            lang=settings.paddle_ocr_lang,
            enable_mkldnn=True,  # Enable MKL-DNN for faster CPU inference
        )
        create_time = time.time() - create_start
        total_time = time.time() - init_start
        logger.info("[PaddleOCR] Instance created in %.2fs (total init: %.2fs)", create_time, total_time)
    else:
        logger.debug("[PaddleOCR] Using cached instance")
    
    return _ocr_instance


def ocr_pdf(path: str | Path) -> List[PageData]:
    """
    Process a PDF through PaddleOCR and return PageData.
    
    Args:
        path: Path to PDF file
    
    Returns:
        List of PageData objects with extracted text blocks
    """
    import time
    total_start = time.time()
    path = Path(path)
    logger.info("[PaddleOCR] Running OCR on PDF: %s", path)
    
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for OCR rendering. Install via `pip install PyMuPDF`."
        ) from exc
    
    # Get cached OCR instance (avoids slow model reloading)
    logger.debug("[PaddleOCR] Getting OCR instance...")
    ocr_start = time.time()
    ocr = _get_ocr()
    logger.debug("[PaddleOCR] Got OCR instance in %.2fs", time.time() - ocr_start)
    
    doc = fitz.open(path)
    pages: List[PageData] = []
    total_pages = len(doc)
    logger.info("[PaddleOCR] Processing %d pages...", total_pages)
    
    for page in doc:
        page_start = time.time()
        page_num = page.number + 1
        logger.debug("[PaddleOCR] Page %d/%d: Rendering...", page_num, total_pages)
        
        # Render page at moderate resolution for OCR (150 DPI is sufficient for OCR)
        # Higher DPI = much slower processing (300 DPI + 2x matrix = ~144 seconds per page!)
        render_start = time.time()
        pix = page.get_pixmap(dpi=150)
        logger.debug("[PaddleOCR] Page %d: Rendered in %.2fs (%dx%d)", 
                    page_num, time.time() - render_start, pix.width, pix.height)
        
        # Convert pixmap to numpy array for PaddleOCR
        import numpy as np
        from PIL import Image
        
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_array = np.array(img)
        
        # Run OCR using predict() (new 3.x API)
        logger.debug("[PaddleOCR] Page %d: Running OCR...", page_num)
        ocr_start = time.time()
        ocr_result = ocr.predict(img_array)
        ocr_time = time.time() - ocr_start
        logger.debug("[PaddleOCR] Page %d: OCR took %.2fs", page_num, ocr_time)
        
        # Convert PaddleOCR results to TextBlocks
        text_blocks = _paddle_results_to_text_blocks(ocr_result, pix.width, pix.height)
        
        page_data = PageData(
            page_num=page_num,
            width=page.rect.width,
            height=page.rect.height,
            blocks=text_blocks,
        )
        page_data.metadata = {
            "extraction_method": "ocr_paddle",
            "ocr_engine_used": "paddle",
            "dpi": 150,
        }
        pages.append(page_data)
        
        page_time = time.time() - page_start
        logger.info("[PaddleOCR] Page %d/%d: %d blocks in %.2fs", 
                   page_num, total_pages, len(text_blocks), page_time)
    
    doc.close()
    total_time = time.time() - total_start
    total_blocks = sum(len(p.blocks) for p in pages)
    logger.info("[PaddleOCR] Processed %d pages, %d blocks in %.2fs (%.2fs/page)", 
               len(pages), total_blocks, total_time, total_time / max(len(pages), 1))
    return pages


def _paddle_results_to_text_blocks(
    ocr_result: List,
    img_width: float,
    img_height: float
) -> List[TextBlock]:
    """
    Convert PaddleOCR 3.x results to TextBlock format.
    
    PaddleOCR 3.x returns a list of dicts with keys:
    - 'rec_texts': list of recognized texts
    - 'rec_scores': list of confidence scores
    - 'dt_polys': list of polygon coordinates (4 points each)
    
    We convert to our format: {"x": x, "y": y, "width": w, "height": h}
    """
    text_blocks = []
    
    if not ocr_result:
        return text_blocks
    
    # PaddleOCR 3.x returns a list of result dicts (one per page/image)
    for result in ocr_result:
        if not isinstance(result, dict):
            continue
            
        rec_texts = result.get('rec_texts', [])
        rec_scores = result.get('rec_scores', [])
        dt_polys = result.get('dt_polys', [])
        
        # Iterate through all detected text regions
        for i, text in enumerate(rec_texts):
            if not text or not text.strip():
                continue
            
            confidence = rec_scores[i] if i < len(rec_scores) else 1.0
            
            # Get polygon coordinates
            if i < len(dt_polys):
                polygon = dt_polys[i]
                # dt_polys contains 4 corner points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                x_coords = [point[0] for point in polygon]
                y_coords = [point[1] for point in polygon]
                
                x_min = min(x_coords)
                y_min = min(y_coords)
                x_max = max(x_coords)
                y_max = max(y_coords)
                
                bbox = {
                    "x": float(x_min),
                    "y": float(y_min),
                    "width": float(x_max - x_min),
                    "height": float(y_max - y_min),
                }
            else:
                # Fallback if no polygon available
                bbox = {"x": 0, "y": 0, "width": 0, "height": 0}
            
            # Create TextBlock with metadata
            block = TextBlock(
                text=text.strip(),
                bbox=bbox,
                style=None,
                metadata={
                    "ocr_engine": "paddle",
                    "bbox_source": "exact",
                    "confidence": float(confidence),
                }
            )
            text_blocks.append(block)
    
    return text_blocks
