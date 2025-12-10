"""Input validation helpers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

SUPPORTED_EXTENSIONS = {".pdf"}
MAX_PAGES = 120  # hard cap to avoid runaway processing


def validate_pdf_path(path: str | os.PathLike) -> Path:
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise ValueError(f"File not found: {pdf_path}")
    if pdf_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {pdf_path.suffix}")
    return pdf_path


def validate_page_range(start: int, end: int) -> Tuple[int, int]:
    if start < 0 or end < 0 or end < start:
        raise ValueError("Invalid page range")
    if end - start > MAX_PAGES:
        raise ValueError("Requested page range exceeds maximum allowed")
    return start, end


def validate_bbox_variety(bboxes: List[Dict[str, float]], min_variation: float = 0.01) -> bool:
    """
    Validate that bounding boxes have varied sizes and positions.
    
    Args:
        bboxes: List of bounding boxes in {"x": x, "y": y, "width": w, "height": h} format
        min_variation: Minimum variation threshold (default 0.01 = 1% of page size)
    
    Returns:
        True if bboxes have sufficient variety, False otherwise
    """
    if len(bboxes) < 2:
        return True  # Single bbox or empty list is considered valid
    
    # Extract all x, y, width, height values
    x_values = [bbox.get("x", 0.0) for bbox in bboxes]
    y_values = [bbox.get("y", 0.0) for bbox in bboxes]
    widths = [bbox.get("width", 0.0) for bbox in bboxes]
    heights = [bbox.get("height", 0.0) for bbox in bboxes]
    
    # Check if values are within normalized range [0.0, 1.0]
    for bbox in bboxes:
        x = bbox.get("x", 0.0)
        y = bbox.get("y", 0.0)
        width = bbox.get("width", 0.0)
        height = bbox.get("height", 0.0)
        
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return False
        if not (0.0 <= width <= 1.0 and 0.0 <= height <= 1.0):
            return False
        if x + width > 1.0 or y + height > 1.0:
            return False
    
    # Check for variety in x positions
    x_range = max(x_values) - min(x_values)
    if x_range < min_variation:
        return False
    
    # Check for variety in y positions
    y_range = max(y_values) - min(y_values)
    if y_range < min_variation:
        return False
    
    # Check for variety in widths
    width_range = max(widths) - min(widths)
    if width_range < min_variation:
        return False
    
    # Check for variety in heights
    height_range = max(heights) - min(heights)
    if height_range < min_variation:
        return False
    
    return True
