"""OCR visualization utilities for drawing bounding boxes on images."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from utils.coordinates import bbox_tuple_to_dict, clamp_bbox_dict, denormalize_bbox

if TYPE_CHECKING:
    from comparison.models import PageData


def _filter_contained_blocks(blocks: list) -> list:
    """
    Filter out blocks that strictly contain other blocks (e.g. table wrappers).
    Also filters duplicates.
    """
    if not blocks:
        return []
        
    n = len(blocks)
    to_remove = set()
    
    # Pre-calculate geometries (x, y, x2, y2, area)
    geoms = []
    for b in blocks:
        # b can be TextBlock object or dict
        if isinstance(b, dict):
             bx = b.get("bbox", {})
        else:
             bx = getattr(b, "bbox", {})
             
        x, y = float(bx.get('x', 0)), float(bx.get('y', 0))
        w, h = float(bx.get('width', 0)), float(bx.get('height', 0))
        geoms.append((x, y, x + w, y + h, w * h))

    for i in range(n):
        if i in to_remove:
            continue
            
        xi1, yi1, xi2, yi2, area_i = geoms[i]
        
        for j in range(n):
            if i == j:
                continue
            if j in to_remove:
                continue
                
            xj1, yj1, xj2, yj2, area_j = geoms[j]
            
            # Intersection
            xx1 = max(xi1, xj1)
            yy1 = max(yi1, yj1)
            xx2 = min(xi2, xj2)
            yy2 = min(yi2, yj2)
            
            w_int = max(0.0, xx2 - xx1)
            h_int = max(0.0, yy2 - yy1)
            area_int = w_int * h_int
            
            if area_int <= 0:
                continue
                
            # Coverage relative to j (the smaller/contained one)
            coverage_j = area_int / area_j if area_j > 0 else 0.0
            
            # If j is effectively inside i
            if coverage_j > 0.90:
                # If i is significantly larger (container) - e.g. table wrapper
                if area_i > 1.2 * area_j:
                    to_remove.add(i)
                    break 
                # If similar size (duplicate)
                else:
                    coverage_i = area_int / area_i if area_i > 0 else 0.0
                    if coverage_i > 0.90:
                         to_remove.add(j)
        
    return [b for k, b in enumerate(blocks) if k not in to_remove]


def _maybe_denormalize_bbox_dict(
    bbox: dict,
    page_width: float,
    page_height: float,
) -> dict:
    """If bbox looks normalized (0..1), convert it to absolute page-point units."""
    if not bbox or page_width <= 0 or page_height <= 0:
        return bbox

    try:
        x = float(bbox.get("x", 0.0))
        y = float(bbox.get("y", 0.0))
        w = float(bbox.get("width", 0.0))
        h = float(bbox.get("height", 0.0))
    except Exception:
        return bbox

    # Heuristic: normalized bboxes should be within [0,1] with size also <= 1.
    if -0.01 <= x <= 1.01 and -0.01 <= y <= 1.01 and 0.0 <= w <= 1.01 and 0.0 <= h <= 1.01:
        x0, y0, x1, y1 = denormalize_bbox({"x": x, "y": y, "width": w, "height": h}, page_width, page_height)
        return clamp_bbox_dict(bbox_tuple_to_dict((x0, y0, x1, y1)), page_width, page_height)

    return bbox


def draw_ocr_bboxes(
    image: Image.Image,
    page_data: "PageData",
    show_text: bool = True,
    show_confidence: bool = False,
    box_color: Tuple[int, int, int] = (0, 255, 0),
    text_color: Tuple[int, int, int] = (255, 0, 0),
    line_width: int = 2,
) -> Image.Image:
    """
    Draw bounding boxes on an image based on OCR results.

    Args:
        image: PIL Image to draw on (a copy is made).
        page_data: PageData object with blocks containing bboxes.
        show_text: If True, draw the detected text near the box.
        show_confidence: If True, append confidence score to the text.
        box_color: RGB tuple for bounding box color.
        text_color: RGB tuple for text label color.
        line_width: Width of the bounding box lines.

    Returns:
        A new PIL Image with bounding boxes drawn.
    """
    # Work on a copy
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    # Try to load a basic font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        font = ImageFont.load_default()

    scale_x = image.width / page_data.width if page_data.width > 0 else 1.0
    scale_y = image.height / page_data.height if page_data.height > 0 else 1.0

    # Filter out container blocks (like full table wrappers) that obscure content
    blocks_to_draw = _filter_contained_blocks(page_data.blocks)

    for block in blocks_to_draw:
        # Check for multiple bboxes in metadata (for collapsed logical changes)
        bboxes_to_draw = block.metadata.get("bboxes", [block.bbox]) if block.metadata else [block.bbox]
        if not bboxes_to_draw:
            bboxes_to_draw = [block.bbox]
        
        for bbox in bboxes_to_draw:
            if not bbox:
                continue

            bbox = _maybe_denormalize_bbox_dict(bbox, float(page_data.width), float(page_data.height))

            # Scale coordinates from PageData points (72 DPI) to Image pixels
            x = bbox.get("x", 0) * scale_x
            y = bbox.get("y", 0) * scale_y
            w = bbox.get("width", 0) * scale_x
            h = bbox.get("height", 0) * scale_y

            # Draw rectangle
            draw.rectangle(
                [(x, y), (x + w, y + h)],
                outline=box_color,
                width=line_width,
            )

        # Draw text label once (near the first bbox)
        if show_text:
            first_bbox = bboxes_to_draw[0] if bboxes_to_draw else block.bbox
            if first_bbox:
                first_bbox = _maybe_denormalize_bbox_dict(first_bbox, float(page_data.width), float(page_data.height))
                x = first_bbox.get("x", 0) * scale_x
                y = first_bbox.get("y", 0) * scale_y
                h = first_bbox.get("height", 0) * scale_y
                
                label = block.text
                if show_confidence:
                    conf = block.metadata.get("confidence", "N/A") if block.metadata else "N/A"
                    label = f"{label} ({conf})"
                
                # Draw text above the box
                text_y = y - 14 if y > 14 else y + h + 2
                draw.text((x, text_y), label, fill=text_color, font=font)

    return img_copy


def visualize_ocr_on_pdf_page(
    pdf_path: str,
    page_num: int = 0,
    ocr_engine: str = "paddle",
    output_path: Optional[str] = None,
    show_text: bool = True,
    show_confidence: bool = False,
) -> Image.Image:
    """
    Run OCR on a specific PDF page and visualize bounding boxes.

    Args:
        pdf_path: Path to the PDF file.
        page_num: Page number (0-indexed).
        ocr_engine: OCR engine to use ("paddle" or "tesseract").
        output_path: If provided, save the result image to this path.
        show_text: If True, draw the detected text near the box.
        show_confidence: If True, append confidence score to the text.

    Returns:
        PIL Image with bounding boxes drawn.
    """
    import fitz  # PyMuPDF
    from pathlib import Path

    # Open PDF and render the page at 300 DPI for visualization
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=300)  # Removed Matrix(2,2) to fix scaling
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()

    # Run OCR
    if ocr_engine == "paddle":
        from extraction.paddle_ocr_engine import ocr_pdf
    elif ocr_engine == "tesseract":
        from extraction.tesseract_ocr_engine import ocr_pdf
    else:
        raise ValueError(f"Unknown OCR engine: {ocr_engine}")

    pages = ocr_pdf(pdf_path)
    if not pages or page_num >= len(pages):
        raise ValueError(f"OCR returned no data for page {page_num}")

    page_data = pages[page_num]

    # Draw bounding boxes
    result_img = draw_ocr_bboxes(
        img, page_data, show_text=show_text, show_confidence=show_confidence
    )

    # Save if output path provided
    if output_path:
        result_img.save(output_path)

    return result_img
