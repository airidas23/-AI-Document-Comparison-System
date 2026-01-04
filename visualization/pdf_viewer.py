"""Side-by-side PDF rendering component."""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import numpy as np

from comparison.models import Diff, PageData
from utils.logging import logger
from visualization.diff_renderer import overlay_diffs, overlay_heatmap


def render_pages(
    pdf_path: str | Path,
    dpi: int = 144,
    diffs: List[Diff] | None = None,
    page_data: PageData | None = None,
    scale_factor: float = 2.0,
    doc_side: str = "a",
) -> List[Tuple[int, np.ndarray]]:
    """
    Render PDF pages to images for Gradio display with high-DPI support.
    
    Uses normalized coordinates (0-1) for bounding boxes to ensure highlights
    remain accurate regardless of display size, similar to React example.
    
    Args:
        pdf_path: Path to PDF file
        dpi: Base resolution for rendering
        diffs: Optional list of diffs to highlight on pages
        page_data: Optional page data for coordinate scaling
        scale_factor: Scale factor for high-DPI rendering (default 2.0 for crisp text)
    
    Returns:
        List of (page_num, numpy_image_array) tuples
    """
    pdf_path = Path(pdf_path)
    logger.info("Rendering PDF for display: %s (DPI=%d, scale=%.1f)", pdf_path, dpi, scale_factor)
    
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for PDF rendering. Install via `pip install PyMuPDF`."
        ) from exc
    
    doc = fitz.open(pdf_path)
    rendered = []
    
    # Group diffs by page.
    # Note: For Document B rendering we should prefer side-specific page numbering
    # (page_num_b) when available.
    diffs_by_page: dict[int, List[Diff]] = {}
    page_dimensions: dict[int, Tuple[float, float]] = {}
    
    if diffs:
        side = str(doc_side).lower()
        use_b = side.startswith("b")
        for diff in diffs:
            key_page = diff.page_num
            if use_b:
                key_page = getattr(diff, "page_num_b", None) or diff.page_num
            if key_page not in diffs_by_page:
                diffs_by_page[key_page] = []
            diffs_by_page[key_page].append(diff)
    
    for page in doc:
        page_num = page.number + 1
        page_width = page.rect.width
        page_height = page.rect.height
        page_dimensions[page_num] = (page_width, page_height)
        
        # High-DPI rendering: scale factor 2 for crisp text (matching React example)
        matrix = fitz.Matrix(scale_factor, scale_factor)
        pix = page.get_pixmap(dpi=int(dpi * scale_factor), matrix=matrix)

        # Convert to numpy array
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        
        # Convert RGBA to RGB if needed (Gradio Gallery expects RGB)
        if pix.n == 4:  # RGBA
            # Convert RGBA to RGB by dropping alpha channel
            img = img[:, :, :3]
        elif pix.n == 1:  # Grayscale
            # Convert grayscale to RGB
            img = np.stack([img[:, :, 0]] * 3, axis=2)
        
        # Ensure image is contiguous in memory for Gradio
        img = np.ascontiguousarray(img)

        # Diffs should already have normalized bbox coordinates
        if page_num in diffs_by_page:
            page_diffs = diffs_by_page[page_num]
            
            # Ensure page dimensions are in metadata
            for diff in page_diffs:
                if "page_width" not in diff.metadata:
                    diff.metadata["page_width"] = page_width
                if "page_height" not in diff.metadata:
                    diff.metadata["page_height"] = page_height
            
            # Overlay diffs using normalized coordinates
            # Pass actual rendered image dimensions for coordinate conversion
            img = overlay_diffs(
                img,
                page_diffs,
                page_width,
                page_height,
                use_normalized=True,
                doc_side="b" if str(doc_side).lower().startswith("b") else "a",
            )

        rendered.append((page_num, img))
    
    doc.close()
    logger.debug("Rendered %d pages with normalized coordinates", len(rendered))
    return rendered


def render_page_pair(
    pdf_a_path: str | Path,
    pdf_b_path: str | Path,
    page_num: int,
    dpi: int = 144,
    diffs: List[Diff] | None = None,
    scale_factor: float = 2.0,
) -> Tuple[bytes, bytes]:
    """
    Render a specific page pair for side-by-side display with high-DPI support.
    
    Returns:
        Tuple of (image_a_bytes, image_b_bytes)
    """
    pdf_a_path = Path(pdf_a_path)
    pdf_b_path = Path(pdf_b_path)
    
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError("PyMuPDF is required")
    
    doc_a = fitz.open(pdf_a_path)
    doc_b = fitz.open(pdf_b_path)
    
    # Get pages (0-indexed)
    page_idx = page_num - 1
    matrix = fitz.Matrix(scale_factor, scale_factor)
    
    if page_idx < len(doc_a):
        page_a = doc_a[page_idx]
        pix_a = page_a.get_pixmap(dpi=int(dpi * scale_factor), matrix=matrix)
        img_a_bytes = pix_a.tobytes("png")
    else:
        img_a_bytes = b""
    
    if page_idx < len(doc_b):
        page_b = doc_b[page_idx]
        pix_b = page_b.get_pixmap(dpi=int(dpi * scale_factor), matrix=matrix)
        img_b_bytes = pix_b.tobytes("png")
    else:
        img_b_bytes = b""
    
    doc_a.close()
    doc_b.close()
    
    return (img_a_bytes, img_b_bytes)


def build_pdf_gallery(label: str, height: int = 600) -> gr.Gallery:
    """Build a Gradio Gallery component for PDF display."""
    return gr.Gallery(
        label=label,
        columns=1,
        height=height,
        show_label=True,
        type="numpy",
    )
