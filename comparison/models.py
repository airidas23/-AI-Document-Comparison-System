"""Shared data models for extraction and comparison."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

from utils.coordinates import bbox_dict_to_tuple, bbox_tuple_to_dict, denormalize_bbox, normalize_bbox


@dataclass
class Style:
    font: Optional[str] = None
    size: Optional[float] = None
    bold: bool = False
    italic: bool = False
    color: Optional[Tuple[int, int, int]] = None  # RGB
    
    def get_fingerprint(self) -> dict:
        """
        Get normalized style fingerprint for deterministic comparison.
        
        Returns:
            Dictionary with normalized style attributes:
            - font_family_normalized: normalized font name
            - weight: "bold" or "regular"
            - slant: "italic" or "normal"
            - size_bucket: rounded font size
        """
        from utils.style_normalization import normalize_font_name, normalize_font_size
        
        return {
            "font_family_normalized": normalize_font_name(self.font or ""),
            "weight": "bold" if self.bold else "regular",
            "slant": "italic" if self.italic else "normal",
            "size_bucket": normalize_font_size(self.size or 0.0),
        }


@dataclass
class TextBlock:
    text: str
    bbox: Dict[str, float]  # {"x": x, "y": y, "width": w, "height": h} in absolute coordinates
    style: Style | None = None
    metadata: dict = field(default_factory=dict)  # OCR engine, bbox_source, etc.
    
    def normalize_bbox(self, page_width: float, page_height: float) -> Dict[str, float]:
        """Convert bbox to normalized coordinates (0-1)."""
        # Convert dict to tuple for normalization, then back to dict
        bbox_tuple = bbox_dict_to_tuple(self.bbox)
        return normalize_bbox(bbox_tuple, page_width, page_height)
    
    def denormalize_bbox(self, page_width: float, page_height: float) -> Tuple[float, float, float, float]:
        """Convert normalized bbox back to absolute coordinates."""
        return denormalize_bbox(self.bbox, page_width, page_height)
    
    def get_bbox_tuple(self) -> Tuple[float, float, float, float]:
        """Get bbox as (x0, y0, x1, y1) tuple for compatibility."""
        return bbox_dict_to_tuple(self.bbox)


@dataclass
class PageData:
    page_num: int
    width: float
    height: float
    blocks: List[TextBlock] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


DiffType = Literal["added", "deleted", "modified"]
ChangeType = Literal["content", "formatting", "layout", "visual"]


@dataclass
class Diff:
    page_num: int
    diff_type: DiffType
    change_type: ChangeType
    old_text: Optional[str]
    new_text: Optional[str]
    bbox: Optional[Dict[str, float]] = None  # {"x": x, "y": y, "width": w, "height": h} in normalized coordinates
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)
    
    def normalize_bbox(self, page_width: float, page_height: float) -> Optional[Dict[str, float]]:
        """Convert bbox to normalized coordinates (0-1). Returns None if bbox is None."""
        if self.bbox is None:
            return None
        # If bbox is already in dict format, check if it needs normalization
        # Assume bbox is already normalized if it's a dict
        if isinstance(self.bbox, dict):
            # Validate it's normalized (values should be in [0, 1])
            x = self.bbox.get("x", 0.0)
            y = self.bbox.get("y", 0.0)
            width = self.bbox.get("width", 0.0)
            height = self.bbox.get("height", 0.0)
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= width <= 1.0 and 0.0 <= height <= 1.0:
                return self.bbox
            # Otherwise normalize it
        # Convert tuple to dict if needed, then normalize
        if isinstance(self.bbox, tuple):
            bbox_tuple = self.bbox
        else:
            bbox_tuple = bbox_dict_to_tuple(self.bbox)
        return normalize_bbox(bbox_tuple, page_width, page_height)
    
    def denormalize_bbox(self, page_width: float, page_height: float) -> Optional[Tuple[float, float, float, float]]:
        """Convert normalized bbox back to absolute coordinates. Returns None if bbox is None."""
        if self.bbox is None:
            return None
        return denormalize_bbox(self.bbox, page_width, page_height)
    
    def get_normalized_bbox(self, page_width: float, page_height: float) -> Optional[Dict[str, float]]:
        """
        Get normalized bbox, storing page dimensions in metadata for later denormalization.
        This ensures coordinates remain accurate regardless of display size.
        """
        if self.bbox is None:
            return None
        
        # Store page dimensions in metadata if not already present
        if "page_width" not in self.metadata:
            self.metadata["page_width"] = page_width
        if "page_height" not in self.metadata:
            self.metadata["page_height"] = page_height
        
        return self.normalize_bbox(page_width, page_height)
    
    def get_bbox_tuple(self) -> Optional[Tuple[float, float, float, float]]:
        """Get bbox as (x0, y0, x1, y1) tuple for compatibility."""
        if self.bbox is None:
            return None
        if isinstance(self.bbox, tuple):
            return self.bbox
        return bbox_dict_to_tuple(self.bbox)


@dataclass
class ComparisonResult:
    doc1: str
    doc2: str
    pages: List[PageData] = field(default_factory=list)
    diffs: List[Diff] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
