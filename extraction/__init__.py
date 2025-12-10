"""PDF extraction module with auto-detection for digital vs scanned documents."""
from __future__ import annotations

from pathlib import Path
from typing import List

from comparison.models import PageData
from config.settings import settings
from extraction.ocr_router import ocr_pdf_multi
from extraction.pdf_parser import parse_pdf
from utils.logging import logger


def extract_pdf(path: str | Path, force_ocr: bool = False) -> List[PageData]:
    """
    Extract text and layout from a PDF, auto-detecting if it's digital or scanned.
    
    Supports OCR enhancement modes:
    - 'auto': Use OCR only for scanned PDFs (default)
    - 'hybrid': Use both native extraction and OCR, merge results
    - 'ocr_only': Force OCR for all documents
    
    Args:
        path: Path to the PDF file
        force_ocr: If True, always use OCR regardless of detection
    
    Returns:
        List of PageData objects with extracted text blocks
    """
    path = Path(path)
    logger.info("Extracting PDF: %s (force_ocr=%s)", path, force_ocr)
    
    # Check OCR enhancement mode
    ocr_mode = getattr(settings, 'ocr_enhancement_mode', 'auto')
    use_ocr_for_all = getattr(settings, 'use_ocr_for_all_documents', False)
    
    if force_ocr or use_ocr_for_all or ocr_mode == 'ocr_only':
        logger.info("Using OCR mode (force_ocr=%s, use_ocr_for_all=%s, mode=%s)", 
                   force_ocr, use_ocr_for_all, ocr_mode)
        return ocr_pdf_multi(path)
    
    # Auto-detect: check if PDF has extractable text
    is_scanned = _is_scanned_pdf(path)
    
    if is_scanned:
        logger.info("Detected scanned PDF, using OCR")
        return ocr_pdf_multi(path)
    elif ocr_mode == 'hybrid':
        # Hybrid mode: extract with native parser, then enhance with OCR
        logger.info("Using hybrid mode: native extraction + OCR enhancement")
        native_pages = parse_pdf(path)
        ocr_pages = ocr_pdf_multi(path)
        
        # Merge results: prefer OCR text for better accuracy, keep native formatting
        merged_pages = _merge_extraction_results(native_pages, ocr_pages)
        return merged_pages
    else:
        logger.info("Detected digital PDF, using text extraction")
        return parse_pdf(path)


def _merge_extraction_results(
    native_pages: List[PageData],
    ocr_pages: List[PageData],
) -> List[PageData]:
    """
    Merge native PDF extraction with OCR results.
    
    Combines the best of both: OCR text accuracy with native formatting/style info.
    
    Args:
        native_pages: Pages extracted using native PDF parsing
        ocr_pages: Pages extracted using OCR
    
    Returns:
        Merged PageData objects
    """
    merged = []
    
    # Create lookup for OCR pages
    ocr_lookup = {page.page_num: page for page in ocr_pages}
    
    for native_page in native_pages:
        merged_page = PageData(
            page_num=native_page.page_num,
            width=native_page.width,
            height=native_page.height,
            blocks=native_page.blocks.copy(),
            metadata=native_page.metadata.copy(),
        )
        
        # If OCR page exists for this page number, merge OCR text blocks
        if native_page.page_num in ocr_lookup:
            ocr_page = ocr_lookup[native_page.page_num]
            
            # Add OCR blocks (they may have better text accuracy)
            # Keep native blocks for formatting info, but prefer OCR text
            if ocr_page.blocks:
                # Merge strategy: use OCR text if native extraction has little/no text
                native_text_length = sum(len(b.text) for b in native_page.blocks)
                ocr_text_length = sum(len(b.text) for b in ocr_page.blocks)
                
                if ocr_text_length > native_text_length * 1.2:
                    # OCR found significantly more text, prefer OCR blocks
                    logger.debug(
                        "Page %d: OCR found more text (%d vs %d chars), using OCR blocks",
                        native_page.page_num, ocr_text_length, native_text_length
                    )
                    merged_page.blocks = ocr_page.blocks.copy()
                else:
                    # Native extraction is good, but add OCR blocks as supplement
                    merged_page.blocks.extend(ocr_page.blocks)
            
            # Merge metadata
            merged_page.metadata.update({
                "extraction_method": "hybrid",
                "native_blocks": len(native_page.blocks),
                "ocr_blocks": len(ocr_page.blocks),
            })
            merged_page.metadata.update(ocr_page.metadata)
        
        merged.append(merged_page)
    
    return merged


def _is_scanned_pdf(path: Path, threshold: float = 0.1) -> bool:
    """
    Detect if a PDF is scanned (image-based) by checking text content.
    
    Args:
        path: Path to PDF file
        threshold: Minimum ratio of pages with text to consider it digital
    
    Returns:
        True if PDF appears to be scanned (low text content)
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not available, assuming digital PDF")
        return False
    
    doc = fitz.open(path)
    pages_with_text = 0
    total_pages = len(doc)
    
    # Sample first few pages to determine type
    sample_size = min(5, total_pages)
    for i in range(sample_size):
        page = doc[i]
        text = page.get_text().strip()
        if len(text) > 50:  # At least 50 characters of text
            pages_with_text += 1
    
    doc.close()
    
    text_ratio = pages_with_text / sample_size if sample_size > 0 else 0
    is_scanned = text_ratio < threshold
    
    logger.debug(
        "PDF detection: %d/%d pages have text (ratio=%.2f), scanned=%s",
        pages_with_text,
        sample_size,
        text_ratio,
        is_scanned,
    )
    
    return is_scanned

