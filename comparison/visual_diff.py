"""Pixel-level comparison and heatmap generation."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from utils.logging import logger


def generate_heatmap(
    pdf_a: str | Path,
    pdf_b: str | Path,
    dpi: int | None = None,
    threshold: int | None = None,
) -> List[Tuple[int, np.ndarray]]:
    """
    Generate heatmap images showing pixel-level differences between PDFs.
    
    Args:
        pdf_a: Path to first PDF
        pdf_b: Path to second PDF
        dpi: Resolution for rendering (lower = faster). If None, uses settings.
        threshold: Pixel difference threshold for noise filtering. If None, uses settings.
    
    Returns:
        List of (page_num, heatmap_image) tuples
    """
    from config.settings import settings
    
    if dpi is None:
        dpi = settings.visual_diff_dpi
    if threshold is None:
        threshold = settings.visual_diff_pixel_threshold
    
    logger.info("Generating visual diff heatmap for %s vs %s", pdf_a, pdf_b)
    
    try:
        import cv2
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV and PyMuPDF are required for visual diff. Install via `pip install opencv-python PyMuPDF`."
        ) from exc
    
    pdf_a = Path(pdf_a)
    pdf_b = Path(pdf_b)
    
    doc_a = fitz.open(pdf_a)
    doc_b = fitz.open(pdf_b)
    
    heatmaps: List[Tuple[int, np.ndarray]] = []
    max_pages = min(len(doc_a), len(doc_b))
    
    for page_num in range(max_pages):
        page_a = doc_a[page_num]
        page_b = doc_b[page_num]
        
        # Render pages to images
        pix_a = page_a.get_pixmap(dpi=dpi, matrix=fitz.Matrix(1, 1))
        pix_b = page_b.get_pixmap(dpi=dpi, matrix=fitz.Matrix(1, 1))
        
        # Convert to numpy arrays
        img_a = np.frombuffer(pix_a.samples, dtype=np.uint8).reshape(
            pix_a.height, pix_a.width, pix_a.n
        )
        img_b = np.frombuffer(pix_b.samples, dtype=np.uint8).reshape(
            pix_b.height, pix_b.width, pix_b.n
        )
        
        # Check for empty images
        if img_a.size == 0 or img_b.size == 0:
            logger.warning("Empty image for visual diff page %d", page_num + 1)
            continue
        
        # Resize to same dimensions if needed
        if img_a.shape != img_b.shape:
            target_h = max(img_a.shape[0], img_b.shape[0])
            target_w = max(img_a.shape[1], img_b.shape[1])
            img_a = cv2.resize(img_a, (target_w, target_h))
            img_b = cv2.resize(img_b, (target_w, target_h))
        
        # Convert to grayscale for comparison
        if len(img_a.shape) == 3:
            gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
        else:
            gray_a = img_a
        
        if len(img_b.shape) == 3:
            gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
        else:
            gray_b = img_b
        
        # Compute absolute difference
        diff = cv2.absdiff(gray_a, gray_b)
        
        # Apply threshold to filter noise
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Create heatmap: convert difference to color-coded image
        heatmap = _create_heatmap_image(diff, thresh)
        
        heatmaps.append((page_num + 1, heatmap))
    
    doc_a.close()
    doc_b.close()
    
    logger.info("Generated %d heatmaps", len(heatmaps))
    return heatmaps


def _create_heatmap_image(diff: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create a color-coded heatmap from difference image."""
    import cv2
    
    # Normalize difference to 0-255 range
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply colormap (JET: blue=low, red=high)
    heatmap = cv2.applyColorMap(diff_norm.astype(np.uint8), cv2.COLORMAP_JET)
    
    # Apply mask to highlight only significant differences
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    heatmap = cv2.bitwise_and(heatmap, mask_3d)
    
    return heatmap


def generate_heatmap_bytes(
    pdf_a: str | Path,
    pdf_b: str | Path,
    dpi: int | None = None,
) -> List[Tuple[int, bytes]]:
    """
    Generate heatmap images as PNG bytes for display in UI.
    
    Args:
        pdf_a: Path to first PDF
        pdf_b: Path to second PDF
        dpi: Resolution for rendering. If None, uses settings.
    
    Returns:
        List of (page_num, png_bytes) tuples
    """
    import cv2
    
    heatmaps = generate_heatmap(pdf_a, pdf_b, dpi=dpi, threshold=None)
    result = []
    
    for page_num, heatmap_img in heatmaps:
        # Encode as PNG
        success, buffer = cv2.imencode(".png", heatmap_img)
        if success:
            result.append((page_num, buffer.tobytes()))
        else:
            logger.warning("Failed to encode heatmap for page %d", page_num)
    
    return result
