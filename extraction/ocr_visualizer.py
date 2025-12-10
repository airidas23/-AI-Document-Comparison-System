"""OCR visualization utilities for drawing bounding boxes on images."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from comparison.models import PageData


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

    for block in page_data.blocks:
        bbox = block.bbox
        x = bbox.get("x", 0)
        y = bbox.get("y", 0)
        w = bbox.get("width", 0)
        h = bbox.get("height", 0)

        # Draw rectangle
        draw.rectangle(
            [(x, y), (x + w, y + h)],
            outline=box_color,
            width=line_width,
        )

        # Draw text if enabled
        if show_text:
            label = block.text
            if show_confidence:
                conf = block.metadata.get("confidence", "N/A")
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

    # Open PDF and render the page
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=300, matrix=fitz.Matrix(2, 2))
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
