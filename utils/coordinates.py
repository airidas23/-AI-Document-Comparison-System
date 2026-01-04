"""Coordinate normalization utilities for bounding boxes."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from comparison.models import PageData


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


def xyxy_to_bbox_dict(x0: float, y0: float, x1: float, y1: float) -> Dict[str, float]:
    """Convert [x0,y0,x1,y1] into canonical {x,y,width,height}."""
    # normalize ordering
    left = float(min(x0, x1))
    top = float(min(y0, y1))
    right = float(max(x0, x1))
    bottom = float(max(y0, y1))
    return {
        "x": left,
        "y": top,
        "width": max(0.0, right - left),
        "height": max(0.0, bottom - top),
    }


def clamp_bbox_dict(
    bbox: Dict[str, float],
    page_width: float,
    page_height: float,
) -> Dict[str, float]:
    """Clamp bbox to page bounds. Ensures non-negative width/height."""
    x = float(bbox.get("x", 0.0))
    y = float(bbox.get("y", 0.0))
    w = float(bbox.get("width", 0.0))
    h = float(bbox.get("height", 0.0))

    # Fix NaNs/None-ish
    if not (x == x): x = 0.0
    if not (y == y): y = 0.0
    if not (w == w): w = 0.0
    if not (h == h): h = 0.0

    # Clamp origin
    x = max(0.0, min(x, float(page_width)))
    y = max(0.0, min(y, float(page_height)))

    # Clamp size to remain inside page
    w = max(0.0, min(w, float(page_width) - x))
    h = max(0.0, min(h, float(page_height) - y))

    return {"x": x, "y": y, "width": w, "height": h}


def bbox_any_to_dict(
    bbox: Any,
    page_width: Optional[float] = None,
    page_height: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    """
    Normalize bbox in either canonical dict or list/tuple [x0,y0,x1,y1].
    Returns canonical dict or None if unusable.
    """
    if bbox is None:
        return None

    # Already canonical dict?
    if isinstance(bbox, dict):
        if {"x", "y", "width", "height"}.issubset(bbox.keys()):
            out = {
                "x": float(bbox.get("x", 0.0)),
                "y": float(bbox.get("y", 0.0)),
                "width": float(bbox.get("width", 0.0)),
                "height": float(bbox.get("height", 0.0)),
            }
            if page_width is not None and page_height is not None:
                out = clamp_bbox_dict(out, page_width, page_height)
            return out

        # Sometimes dict bbox is xyxy-like
        if {"x0", "y0", "x1", "y1"}.issubset(bbox.keys()):
            out = xyxy_to_bbox_dict(bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"])
            if page_width is not None and page_height is not None:
                out = clamp_bbox_dict(out, page_width, page_height)
            return out

        return None

    # list/tuple xyxy
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            x0, y0, x1, y1 = map(float, bbox)
        except Exception:
            return None
        out = xyxy_to_bbox_dict(x0, y0, x1, y1)

        # If we have bounds, clamp. If bbox looks wildly out of bounds, we still clamp
        # (prevents crashes) but this might indicate pixel-space leaked in.
        if page_width is not None and page_height is not None:
            out = clamp_bbox_dict(out, page_width, page_height)
        return out

    return None


def pixel_bbox_to_pdf_points(
    bbox_px: List[float] | Tuple[float, float, float, float],
    pix_width: int,
    pix_height: int,
    page_width: float,
    page_height: float,
) -> Dict[str, float]:
    """
    Convert pixel bbox [x0,y0,x1,y1] to canonical bbox dict in PDF points.
    """
    if pix_width <= 0 or pix_height <= 0:
        # Defensive: avoid divide-by-zero
        return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}

    x0, y0, x1, y1 = map(float, bbox_px)

    scale_x = float(page_width) / float(pix_width)
    scale_y = float(page_height) / float(pix_height)

    x0_pt = x0 * scale_x
    x1_pt = x1 * scale_x
    y0_pt = y0 * scale_y
    y1_pt = y1 * scale_y

    out = xyxy_to_bbox_dict(x0_pt, y0_pt, x1_pt, y1_pt)
    return clamp_bbox_dict(out, page_width, page_height)


def _ensure_bbox_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Fill bbox meta defaults without overriding existing values."""
    if meta.get("bbox_units") is None:
        meta["bbox_units"] = "pt"
    if meta.get("bbox_space") is None:
        meta["bbox_space"] = "page"
    # bbox_source: do not force; keep engine truth. If missing, assume exact.
    if meta.get("bbox_source") is None:
        meta["bbox_source"] = "exact"
    return meta


def _normalize_bbox_in_obj(
    obj: Any,
    page_width: float,
    page_height: float,
) -> Any:
    """
    Recursively normalize any dict that has a 'bbox' key, and any nested lists/dicts.
    """
    if isinstance(obj, list):
        return [_normalize_bbox_in_obj(x, page_width, page_height) for x in obj]

    if isinstance(obj, dict):
        # If this dict has a bbox, normalize it
        if "bbox" in obj:
            bbox_dict = bbox_any_to_dict(obj.get("bbox"), page_width, page_height)
            if bbox_dict is not None:
                obj["bbox"] = bbox_dict
                _ensure_bbox_meta(obj)

        # Recurse into all values
        for k, v in list(obj.items()):
            obj[k] = _normalize_bbox_in_obj(v, page_width, page_height)
        return obj

    return obj


def normalize_page_bboxes(pages: List["PageData"]) -> List["PageData"]:
    """
    Global bbox contract enforcer.

    Ensures:
    - All TextBlock.bbox are dict {x,y,width,height}
    - All metadata bboxes are dict
    - bbox_units="pt", bbox_space="page" exist where possible
    - bboxes clamped to page bounds
    """
    
    for page in pages:
        page_w = float(getattr(page, "width", 0.0) or 0.0)
        page_h = float(getattr(page, "height", 0.0) or 0.0)

        # Normalize text blocks
        for block in getattr(page, "blocks", []) or []:
            bbox_dict = bbox_any_to_dict(getattr(block, "bbox", None), page_w, page_h)
            if bbox_dict is not None:
                block.bbox = bbox_dict

            # Ensure block metadata
            meta = getattr(block, "metadata", None)
            if meta is None:
                meta = {}
                block.metadata = meta
            _ensure_bbox_meta(meta)

        # Normalize metadata recursively
        page_meta = getattr(page, "metadata", None)
        if page_meta is None:
            page_meta = {}
            page.metadata = page_meta

        page.metadata = _normalize_bbox_in_obj(page_meta, page_w, page_h)

    return pages

