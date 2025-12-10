"""Coordinate normalization utilities for bounding boxes."""
from __future__ import annotations

from typing import Dict, Tuple


def normalize_bbox(
    bbox: Tuple[float, float, float, float],
    page_width: float,
    page_height: float,
) -> Dict[str, float]:
    """
    Convert absolute coordinates to normalized (0-1) range in {x, y, width, height} format.
    
    Args:
        bbox: Bounding box as (x0, y0, x1, y1) in absolute coordinates
        page_width: Width of the page in absolute units
        page_height: Height of the page in absolute units
    
    Returns:
        Normalized bounding box as {"x": x, "y": y, "width": w, "height": h} with values in [0.0, 1.0]
    """
    if page_width <= 0 or page_height <= 0:
        raise ValueError("Page dimensions must be positive")
    
    x0, y0, x1, y1 = bbox
    
    # Normalize coordinates and clamp to [0.0, 1.0] range
    x = max(0.0, min(1.0, x0 / page_width))
    y = max(0.0, min(1.0, y0 / page_height))
    width = max(0.0, min(1.0, (x1 - x0) / page_width))
    height = max(0.0, min(1.0, (y1 - y0) / page_height))
    
    # Ensure width and height don't exceed bounds
    if x + width > 1.0:
        width = 1.0 - x
    if y + height > 1.0:
        height = 1.0 - y
    
    return {"x": x, "y": y, "width": width, "height": height}


def denormalize_bbox(
    normalized_bbox: Dict[str, float],
    page_width: float,
    page_height: float,
) -> Tuple[float, float, float, float]:
    """
    Convert normalized coordinates (0-1) back to absolute coordinates.
    
    Args:
        normalized_bbox: Bounding box as {"x": x, "y": y, "width": w, "height": h} in normalized coordinates [0.0, 1.0]
        page_width: Width of the page in absolute units
        page_height: Height of the page in absolute units
    
    Returns:
        Absolute bounding box as (x0, y0, x1, y1)
    """
    if page_width <= 0 or page_height <= 0:
        raise ValueError("Page dimensions must be positive")
    
    x = max(0.0, min(1.0, normalized_bbox.get("x", 0.0)))
    y = max(0.0, min(1.0, normalized_bbox.get("y", 0.0)))
    width = max(0.0, min(1.0, normalized_bbox.get("width", 0.0)))
    height = max(0.0, min(1.0, normalized_bbox.get("height", 0.0)))
    
    # Ensure width and height don't exceed bounds
    if x + width > 1.0:
        width = 1.0 - x
    if y + height > 1.0:
        height = 1.0 - y
    
    # Convert to absolute coordinates
    x0 = x * page_width
    y0 = y * page_height
    x1 = (x + width) * page_width
    y1 = (y + height) * page_height
    
    return (x0, y0, x1, y1)


def bbox_tuple_to_dict(bbox: Tuple[float, float, float, float]) -> Dict[str, float]:
    """
    Convert bbox from (x0, y0, x1, y1) tuple to {x, y, width, height} dict.
    
    Args:
        bbox: Bounding box as (x0, y0, x1, y1)
    
    Returns:
        Bounding box as {"x": x, "y": y, "width": w, "height": h}
    """
    x0, y0, x1, y1 = bbox
    return {
        "x": x0,
        "y": y0,
        "width": x1 - x0,
        "height": y1 - y0,
    }


def bbox_dict_to_tuple(bbox: Dict[str, float]) -> Tuple[float, float, float, float]:
    """
    Convert bbox from {x, y, width, height} dict to (x0, y0, x1, y1) tuple.
    
    Args:
        bbox: Bounding box as {"x": x, "y": y, "width": w, "height": h}
    
    Returns:
        Bounding box as (x0, y0, x1, y1)
    """
    x = bbox.get("x", 0.0)
    y = bbox.get("y", 0.0)
    width = bbox.get("width", 0.0)
    height = bbox.get("height", 0.0)
    return (x, y, x + width, y + height)

