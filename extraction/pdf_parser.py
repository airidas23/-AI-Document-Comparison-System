"""Digital PDF text/style extraction using PyMuPDF."""
from __future__ import annotations

from pathlib import Path
from typing import List

from comparison.models import PageData, Style, TextBlock
from extraction.layout_analyzer import analyze_layout
from utils.coordinates import bbox_tuple_to_dict
from utils.logging import logger


def parse_pdf(path: str | Path) -> List[PageData]:
    """Extract text, fonts, and positions from a digital PDF."""
    path = Path(path)
    logger.info("Parsing PDF (digital): %s", path)
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for PDF parsing. Install via `pip install PyMuPDF`."
        ) from exc
    
    doc = fitz.open(path)
    pages: List[PageData] = []

    for page in doc:
        width, height = page.rect.width, page.rect.height
        page_data = PageData(page_num=page.number + 1, width=width, height=height)
        
        # Extract text blocks with positions
        blocks = page.get_text("blocks", flags=fitz.TEXT_PRESERVE_LIGATURES)
        for block in blocks:
            x0, y0, x1, y1, text, block_no, block_type = block[:7]
            if block_type != 0 or not text.strip():
                continue
            
            # Convert bbox from tuple to dict format
            bbox_dict = bbox_tuple_to_dict((x0, y0, x1, y1))
            
            # Extract style information from text spans
            style = _extract_style_from_page(page, (x0, y0, x1, y1))
            page_data.blocks.append(
                TextBlock(text=text.strip(), bbox=bbox_dict, style=style)
            )
        
        # Store page metadata
        page_data.metadata = {
            "rotation": page.rotation,
            "cropbox": list(page.cropbox) if page.cropbox else None,
        }
        
        pages.append(page_data)
    
    doc.close()
    
    # Run layout analysis to detect tables and figures
    layout_pages = analyze_layout(path)
    layout_lookup = {p.page_num: p for p in layout_pages}
    
    # Merge layout metadata into extracted pages
    for page_data in pages:
        if page_data.page_num in layout_lookup:
            layout_page = layout_lookup[page_data.page_num]
            page_data.metadata.update(layout_page.metadata)
    
    logger.info("Extracted %d pages from PDF", len(pages))
    return pages


def _extract_style_from_page(page, bbox: tuple) -> Style:
    """Extract font/style information for a text region."""
    try:
        import fitz
    except ImportError:
        return Style()
    
    # Get text spans that overlap with this bbox
    x0, y0, x1, y1 = bbox
    spans = page.get_text("dict", flags=fitz.TEXT_PRESERVE_LIGATURES)
    
    # Find the first span in this region to get style info
    font_name = None
    font_size = None
    is_bold = False
    is_italic = False
    color = None
    
    for block in spans.get("blocks", []):
        if block.get("type") != 0:  # Not text block
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                span_bbox = span.get("bbox", [])
                if len(span_bbox) == 4:
                    sx0, sy0, sx1, sy1 = span_bbox
                    # Check if spans overlap
                    if not (sx1 < x0 or sx0 > x1 or sy1 < y0 or sy0 > y1):
                        font_name = span.get("font", font_name)
                        font_size = span.get("size", font_size)
                        flags = span.get("flags", 0)
                        is_bold = bool(flags & 16)  # Bit 4 indicates bold
                        is_italic = bool(flags & 1)  # Bit 0 indicates italic
                        color = span.get("color")
                        if color and isinstance(color, int):
                            # Convert 24-bit color to RGB
                            r = (color >> 16) & 0xFF
                            g = (color >> 8) & 0xFF
                            b = color & 0xFF
                            color = (r, g, b)
                        break
            if font_name:
                break
        if font_name:
            break
    
    return Style(
        font=font_name,
        size=font_size,
        bold=is_bold,
        italic=is_italic,
        color=color,
    )
