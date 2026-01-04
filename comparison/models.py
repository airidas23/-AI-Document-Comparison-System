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
class Token:
    token_id: str
    bbox: Dict[str, float]  # {"x": x, "y": y, "width": w, "height": h} in absolute coordinates
    text: str
    confidence: float = 1.0
    style: Optional[Style] = None  # Style info from span lookup
    span_index: int = -1  # PyMuPDF span index for style lookup


@dataclass
class WordDiff:
    """Word-level diff information with optional character details."""
    word_a: str
    word_b: str
    bbox_a: Optional[Dict[str, float]] = None
    bbox_b: Optional[Dict[str, float]] = None
    change_type: Literal["same", "modified", "added", "deleted"] = "same"
    char_diffs: List[Tuple[str, int, int, str]] = field(default_factory=list)  # (op, i1, i2, text)


@dataclass
class Line:
    line_id: str
    bbox: Dict[str, float]  # {"x": x, "y": y, "width": w, "height": h} in absolute coordinates
    text: str
    confidence: float = 1.0
    reading_order: int = 0
    tokens: List[Token] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)  # For word-level data, granularity info, etc.


@dataclass
class PageData:
    page_num: int
    width: float
    height: float
    blocks: List[TextBlock] = field(default_factory=list)
    lines: List[Line] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


DiffType = Literal["added", "deleted", "modified"]
ChangeType = Literal["content", "formatting", "layout", "visual"]
LayoutChangeType = Literal["none", "reflow", "move", "resize", "reorder"]
ElementType = Literal["text", "table", "figure", "formula", "header", "footer"]


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
    # Enhanced fields for academic comparison
    page_num_b: Optional[int] = None  # Page number in doc B (if different from A)
    bbox_b: Optional[Dict[str, float]] = None  # Bbox in doc B
    word_diffs: List[WordDiff] = field(default_factory=list)  # Word-level details
    style_a: Optional[Style] = None  # Style in doc A
    style_b: Optional[Style] = None  # Style in doc B
    element_type: ElementType = "text"  # Type of element
    layout_change_type: Optional[LayoutChangeType] = None  # For layout changes
    sources: List[str] = field(default_factory=list)  # Which modules detected this diff
    
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


# === Table Comparison Structures ===

@dataclass
class TableCell:
    """Single cell in a table."""
    row: int
    col: int
    text: str
    bbox: Dict[str, float]
    rowspan: int = 1
    colspan: int = 1
    style: Optional[Style] = None
    confidence: float = 1.0


@dataclass
class CellDiff:
    """Difference between two table cells."""
    row: int
    col: int
    cell_a: Optional[TableCell]
    cell_b: Optional[TableCell]
    change_type: Literal["same", "modified", "added", "deleted"]
    text_similarity: float = 0.0
    style_changed: bool = False


@dataclass
class TableDiff:
    """Complete table comparison result."""
    table_id: str
    bbox_a: Optional[Dict[str, float]]
    bbox_b: Optional[Dict[str, float]]
    cell_diffs: List[CellDiff] = field(default_factory=list)
    rows_a: int = 0
    rows_b: int = 0
    cols_a: int = 0
    cols_b: int = 0
    structure_changed: bool = False
    border_changed: bool = False


# === Figure/Formula Comparison Structures ===

@dataclass
class FormulaRegion:
    """Formula region with optional LaTeX OCR."""
    bbox: Dict[str, float]
    latex: Optional[str] = None
    image_bytes: Optional[bytes] = None
    confidence: float = 0.0


@dataclass
class FigureRegion:
    """Figure region with image data and caption."""
    bbox: Dict[str, float]
    caption: Optional[str] = None
    image_bytes: Optional[bytes] = None
    figure_id: Optional[str] = None


# === Report Structures ===

@dataclass
class ComparisonReport:
    """Full comparison report for export."""
    doc_a_path: str
    doc_b_path: str
    timestamp: str
    
    # Summary
    total_pages_a: int = 0
    total_pages_b: int = 0
    pages_compared: int = 0
    
    # Diff counts by category
    content_changes: int = 0
    formatting_changes: int = 0
    layout_changes: int = 0
    visual_changes: int = 0
    
    # Table-specific
    table_cell_changes: int = 0
    
    # All diffs
    diffs: List[Diff] = field(default_factory=list)
    table_diffs: List[TableDiff] = field(default_factory=list)
    
    # Debug info (when enabled)
    debug: Optional[dict] = None
    
    # Metrics
    comparison_time_seconds: float = 0.0
    extraction_method: str = "auto"
