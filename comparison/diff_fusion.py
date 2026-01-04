"""
Diff fusion module for merging diffs from multiple comparison modules.

Implements triangulation logic to:
1. Deduplicate overlapping diffs from different modules
2. Calculate confidence based on module consensus
3. Merge metadata from multiple sources
4. Assign final change_type based on consensus
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

from comparison.models import ChangeType, Diff, DiffType
from utils.logging import logger


@dataclass
class DiffCluster:
    """A cluster of overlapping diffs from potentially multiple modules."""
    diffs: List[Diff] = field(default_factory=list)
    modules: List[str] = field(default_factory=list)
    
    @property
    def page_num(self) -> int:
        return self.diffs[0].page_num if self.diffs else 0
    
    @property
    def merged_bbox(self) -> Optional[Dict[str, float]]:
        """Compute union bbox of all diffs in cluster."""
        bboxes = [d.bbox for d in self.diffs if d.bbox]
        if not bboxes:
            return None
        
        x_min = min(b.get("x", 0.0) for b in bboxes)
        y_min = min(b.get("y", 0.0) for b in bboxes)
        x_max = max(b.get("x", 0.0) + b.get("width", 0.0) for b in bboxes)
        y_max = max(b.get("y", 0.0) + b.get("height", 0.0) for b in bboxes)
        
        return {
            "x": x_min,
            "y": y_min,
            "width": x_max - x_min,
            "height": y_max - y_min,
        }

    @property
    def merged_bbox_b(self) -> Optional[Dict[str, float]]:
        """Pick a reasonable B-side (doc2) bbox.

        Multiple modules may contribute different bbox granularities (e.g., a tight
        word bbox from line comparison vs a paragraph bbox from semantic comparison).
        For UI highlighting, prefer the tightest non-empty bbox.
        """
        bboxes: List[Dict[str, float]] = []
        for d in self.diffs:
            bb = getattr(d, "bbox_b", None)
            if isinstance(bb, dict):
                bboxes.append(bb)
                continue
            md_bb = (getattr(d, "metadata", None) or {}).get("bbox_b")
            if isinstance(md_bb, dict):
                bboxes.append(md_bb)

        if not bboxes:
            return None

        def area(b: Dict[str, float]) -> float:
            try:
                return max(0.0, float(b.get("width", 0.0))) * max(0.0, float(b.get("height", 0.0)))
            except Exception:
                return float("inf")

        candidates = [b for b in bboxes if area(b) > 0.0]
        if not candidates:
            return None

        return min(candidates, key=area)


def calculate_iou(bbox1: Optional[Dict[str, float]], bbox2: Optional[Dict[str, float]]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bboxes.
    
    Args:
        bbox1: First bbox dict {"x", "y", "width", "height"}
        bbox2: Second bbox dict {"x", "y", "width", "height"}
    
    Returns:
        IoU value between 0.0 and 1.0
    """
    if bbox1 is None or bbox2 is None:
        return 0.0
    
    # Extract coordinates
    x1_min = bbox1.get("x", 0.0)
    y1_min = bbox1.get("y", 0.0)
    x1_max = x1_min + bbox1.get("width", 0.0)
    y1_max = y1_min + bbox1.get("height", 0.0)
    
    x2_min = bbox2.get("x", 0.0)
    y2_min = bbox2.get("y", 0.0)
    x2_max = x2_min + bbox2.get("width", 0.0)
    y2_max = y2_min + bbox2.get("height", 0.0)
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def diffs_overlap(diff1: Diff, diff2: Diff, iou_threshold: float = 0.3) -> bool:
    """
    Check if two diffs overlap significantly.
    
    Uses multiple criteria:
    1. Same page
    2. IoU above threshold OR same text content
    """
    # Must be same page
    if diff1.page_num != diff2.page_num:
        return False
    
    # Check bbox overlap
    iou = calculate_iou(diff1.bbox, diff2.bbox)
    if iou >= iou_threshold:
        return True
    
    # Check text similarity for diffs without bbox or low IoU
    if diff1.old_text and diff2.old_text:
        # Simple text overlap check
        text1 = diff1.old_text.strip().lower()
        text2 = diff2.old_text.strip().lower()
        if text1 == text2 or (len(text1) > 10 and text1 in text2) or (len(text2) > 10 and text2 in text1):
            return True
    
    if diff1.new_text and diff2.new_text:
        text1 = diff1.new_text.strip().lower()
        text2 = diff2.new_text.strip().lower()
        if text1 == text2 or (len(text1) > 10 and text1 in text2) or (len(text2) > 10 and text2 in text1):
            return True
    
    return False


def bbox_contains(outer: Optional[Dict[str, float]], inner: Optional[Dict[str, float]], margin: float = 0.02) -> bool:
    """
    Check if outer bbox contains inner bbox (with margin tolerance).
    
    Args:
        outer: Container bbox
        inner: Contained bbox
        margin: Allowed margin outside container (normalized coordinates)
    
    Returns:
        True if inner is within outer (with margin)
    """
    if outer is None or inner is None:
        return False
    
    ox, oy = outer.get("x", 0.0), outer.get("y", 0.0)
    ow, oh = outer.get("width", 0.0), outer.get("height", 0.0)
    ix, iy = inner.get("x", 0.0), inner.get("y", 0.0)
    iw, ih = inner.get("width", 0.0), inner.get("height", 0.0)
    
    # Check if inner bbox center is within outer bbox (with margin)
    inner_cx = ix + iw / 2
    inner_cy = iy + ih / 2
    
    return (ox - margin <= inner_cx <= ox + ow + margin and
            oy - margin <= inner_cy <= oy + oh + margin)


def _bbox_area(bbox: Optional[Dict[str, float]]) -> float:
    if not bbox:
        return 0.0
    try:
        return max(0.0, float(bbox.get("width", 0.0))) * max(0.0, float(bbox.get("height", 0.0)))
    except Exception:
        return 0.0


def _expand_bbox(bbox: Dict[str, float], margin: float) -> Dict[str, float]:
    """Expand a bbox by a margin (expects normalized coordinates in 0..1).

    This is used to tolerate OCR table detector under-estimation.
    """
    x = float(bbox.get("x", 0.0))
    y = float(bbox.get("y", 0.0))
    w = float(bbox.get("width", 0.0))
    h = float(bbox.get("height", 0.0))
    x2 = x + w
    y2 = y + h

    x = max(0.0, x - margin)
    y = max(0.0, y - margin)
    x2 = min(1.0, x2 + margin)
    y2 = min(1.0, y2 + margin)
    return {"x": x, "y": y, "width": max(0.0, x2 - x), "height": max(0.0, y2 - y)}


def _intersection_area(a: Dict[str, float], b: Dict[str, float]) -> float:
    ax1 = float(a.get("x", 0.0))
    ay1 = float(a.get("y", 0.0))
    ax2 = ax1 + float(a.get("width", 0.0))
    ay2 = ay1 + float(a.get("height", 0.0))

    bx1 = float(b.get("x", 0.0))
    by1 = float(b.get("y", 0.0))
    bx2 = bx1 + float(b.get("width", 0.0))
    by2 = by1 + float(b.get("height", 0.0))

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _maybe_text_length(diff: Diff) -> int:
    t1 = (diff.old_text or "").strip()
    t2 = (diff.new_text or "").strip()
    return max(len(t1), len(t2))


def _maybe_word_count(diff: Diff) -> int:
    t = (diff.new_text or diff.old_text or "").strip()
    if not t:
        return 0
    return len([w for w in t.split() if w])


def _looks_like_table_text(diff: Diff) -> bool:
    t = (diff.new_text or diff.old_text or "")
    if not t:
        return False
    # OCR'd tables often yield pipe-delimited pseudo-rows.
    if t.count("|") >= 3:
        return True
    # Heuristic: many short tokens + digits can also indicate table cells.
    tokens = [x for x in t.replace("|", " ").split() if x]
    if len(tokens) >= 10:
        short = sum(1 for x in tokens if len(x) <= 3)
        digits = sum(1 for x in tokens if any(ch.isdigit() for ch in x))
        if short / max(1, len(tokens)) > 0.55 and digits >= 3:
            return True
    return False


def _should_absorb_into_table(table_diff: Diff, candidate: Diff, *, margin: float) -> bool:
    """Heuristic for absorbing non-table OCR diffs into a table cluster.

    Goal: absorb table-local token/punctuation noise, but avoid swallowing real
    paragraph diffs that happen to be near a table (common in scanned pages).
    """
    if candidate.bbox is None or table_diff.bbox is None:
        return False
    if candidate.page_num != table_diff.page_num:
        return False

    table_bbox = _expand_bbox(table_diff.bbox, margin)
    cand_bbox = candidate.bbox

    # Quick containment check (center-based) with tolerance.
    if not bbox_contains(table_bbox, cand_bbox, margin=0.0):
        return False

    table_area = _bbox_area(table_bbox)
    cand_area = _bbox_area(cand_bbox)
    if table_area <= 0.0 or cand_area <= 0.0:
        return False

    table_like = _looks_like_table_text(candidate)

    # Avoid absorbing very large regions (paragraphs) into a table.
    # For OCR table-like text, allow regions close to the table size.
    if cand_area > table_area * (1.25 if table_like else 0.85):
        return False

    # Require meaningful geometric overlap (not just "near" by margin).
    inter = _intersection_area(table_bbox, cand_bbox)
    if inter <= 0.0:
        return False
    inter_over_cand = inter / max(1e-9, cand_area)
    if inter_over_cand < 0.35:
        return False

    # Avoid absorbing long multi-word sentence diffs into tables.
    if (not table_like) and (_maybe_text_length(candidate) > 80 or _maybe_word_count(candidate) > 12):
        return False

    return True


def cluster_diffs(
    diff_lists: List[Tuple[str, List[Diff]]],
    iou_threshold: float = 0.3,
) -> List[DiffCluster]:
    """
    Cluster overlapping diffs from different modules.
    
    Args:
        diff_lists: List of (module_name, diffs) tuples
        iou_threshold: Minimum IoU to consider diffs overlapping
    
    Returns:
        List of DiffCluster objects
    """
    clusters: List[DiffCluster] = []
    
    # First pass: identify table diffs to use as anchors
    table_clusters: List[DiffCluster] = []
    for module_name, diffs in diff_lists:
        for diff in diffs:
            if diff.metadata.get("type") == "table":
                table_clusters.append(DiffCluster(
                    diffs=[diff],
                    modules=[module_name],
                ))
    
    # Add table clusters first
    clusters.extend(table_clusters)
    
    for module_name, diffs in diff_lists:
        for diff in diffs:
            # Skip table diffs (already added)
            if diff.metadata.get("type") == "table":
                continue
            
            # Check if this diff falls within a table region
            matched_table = None
            for cluster in table_clusters:
                table_diff = cluster.diffs[0]  # Original table diff
                # OCR table detectors can under-estimate table width/height.
                # Use a wider containment margin on OCR pages to absorb
                # table-local punctuation/token noise into the table cluster.
                table_margin = 0.08 if (table_diff.metadata or {}).get("is_ocr") else 0.02
                if _should_absorb_into_table(table_diff, diff, margin=table_margin):
                    matched_table = cluster
                    break
            
            if matched_table:
                # Absorb line diff into table cluster
                matched_table.diffs.append(diff)
                if module_name not in matched_table.modules:
                    matched_table.modules.append(module_name)
                continue
            
            # Find matching non-table cluster
            matched_cluster = None
            for cluster in clusters:
                if cluster in table_clusters:
                    continue  # Don't match against table clusters via IoU
                # Check if diff overlaps with any diff in cluster
                for existing_diff in cluster.diffs:
                    if diffs_overlap(diff, existing_diff, iou_threshold):
                        matched_cluster = cluster
                        break
                if matched_cluster:
                    break
            
            if matched_cluster:
                # Add to existing cluster
                matched_cluster.diffs.append(diff)
                if module_name not in matched_cluster.modules:
                    matched_cluster.modules.append(module_name)
            else:
                # Create new cluster
                clusters.append(DiffCluster(
                    diffs=[diff],
                    modules=[module_name],
                ))
    
    return clusters


def determine_consensus_change_type(diffs: List[Diff]) -> ChangeType:
    """
    Determine change_type based on consensus from multiple diffs.
    
    Priority order: content > formatting > layout > visual
    """
    # Special case: keep table *structural* diffs as layout even if we absorbed
    # token-level OCR diffs (which would otherwise bias consensus to content).
    # However, some structure deltas are semantically content (e.g. added/removed
    # computed columns). Those should not be forced to layout.
    for diff in diffs:
        md = diff.metadata or {}
        if md.get("type") == "table" and md.get("table_change") == "structure":
            if md.get("subtype") == "table_columns_changed":
                continue
            return "layout"

    # Special case: figure numbering / visual-content changes should stay visual.
    # These often co-occur with token-level text diffs (caption text line changes),
    # and the generic priority order would otherwise re-label them as content.
    for diff in diffs:
        md = diff.metadata or {}
        if md.get("type") == "figure" and md.get("figure_change") in {
            "numbering",
            "visual_content",
            "figure_added",
            "figure_deleted",
        }:
            return "visual"

    type_counts: Dict[str, int] = {}
    for diff in diffs:
        ct = diff.change_type
        type_counts[ct] = type_counts.get(ct, 0) + 1
    
    # Priority-based selection (content changes are most important)
    priority_order: List[ChangeType] = ["content", "formatting", "layout", "visual"]
    for change_type in priority_order:
        if type_counts.get(change_type, 0) > 0:
            return change_type
    
    return "content"  # Default fallback


def determine_consensus_diff_type(diffs: List[Diff]) -> DiffType:
    """
    Determine diff_type based on consensus from multiple diffs.
    
    Uses majority voting.
    """
    type_counts: Dict[str, int] = {}
    for diff in diffs:
        dt = diff.diff_type
        type_counts[dt] = type_counts.get(dt, 0) + 1
    
    # Return most common type
    if not type_counts:
        return "modified"
    
    return max(type_counts, key=lambda k: type_counts[k])  # type: ignore


def merge_metadata(diffs: List[Diff], modules: List[str]) -> dict:
    """
    Merge metadata from multiple diffs.
    
    Combines metadata from all diffs, with table diffs taking priority
    for type-related fields. Also adds fusion-specific metadata.
    """
    merged: dict = {}
    table_meta: dict = {}
    absorbed_count = 0
    
    for diff in diffs:
        if diff.metadata:
            # Track table metadata separately (it should take priority)
            if diff.metadata.get("type") == "table":
                table_meta = diff.metadata.copy()
            else:
                # Count absorbed line diffs within table
                if table_meta:
                    absorbed_count += 1
                merged.update(diff.metadata)
    
    # Table metadata takes priority
    if table_meta:
        # Preserve table-specific fields
        merged["type"] = "table"
        merged["table_change"] = table_meta.get("table_change", "content")
        if "old_structure" in table_meta:
            merged["old_structure"] = table_meta["old_structure"]
        if "new_structure" in table_meta:
            merged["new_structure"] = table_meta["new_structure"]
        if absorbed_count > 0:
            merged["absorbed_line_diffs"] = absorbed_count
        # Preserve description from table
        if "description" in table_meta:
            merged["description"] = table_meta["description"]
    
    # Add fusion metadata
    merged["fusion_modules"] = modules
    merged["fusion_count"] = len(modules)
    merged["fusion_diff_count"] = len(diffs)
    
    return merged


def calculate_fused_confidence(cluster: DiffCluster, total_modules: int) -> float:
    """
    Calculate confidence based on triangulation logic.
    
    - 3+ modules agree: 0.95
    - 2 modules agree: 0.85
    - 1 module: base confidence * 0.7 (lower weight for single source)
    """
    num_modules = len(cluster.modules)
    
    if num_modules >= 3:
        return 0.95
    elif num_modules == 2:
        return 0.85
    else:
        # Single module: use average confidence from diffs, scaled down
        avg_conf = sum(d.confidence for d in cluster.diffs) / len(cluster.diffs) if cluster.diffs else 0.5
        return min(0.75, avg_conf * 0.7 + 0.3)  # Floor at 0.3, cap at 0.75


def merge_text_content(diffs: List[Diff]) -> Tuple[Optional[str], Optional[str]]:
    """
    Merge text content from multiple diffs.
    
    Prefers longer text or text from content-type diffs.
    """
    old_texts = [d.old_text for d in diffs if d.old_text]
    new_texts = [d.new_text for d in diffs if d.new_text]
    
    # Choose longest text (usually most complete)
    old_text = max(old_texts, key=len) if old_texts else None
    new_text = max(new_texts, key=len) if new_texts else None
    
    return old_text, new_text


# ============================================================================
# Header/Footer Aggregation
# ============================================================================

_DIGIT_RE = re.compile(r"\d+")


def _canonical_text_for_grouping(text: str | None) -> str:
    """Stable key for grouping: lowercase, collapse spaces, digits -> {#}."""
    if not text:
        return ""
    t = " ".join(text.strip().lower().split())
    return _DIGIT_RE.sub("{#}", t)


def _pretty_template(text: str | None) -> str | None:
    """Human-friendly template: keep original casing/punctuation, digits -> {#}."""
    if text is None:
        return None
    return _DIGIT_RE.sub("{#}", text.strip())


def aggregate_repeating_header_footer_diffs(
    diffs: List[Diff],
    *,
    min_pages: int = 2,
) -> List[Diff]:
    """
    Collapse per-page header/footer diffs into a small number of global 'events'.

    Example outcome:
      - 20x "Header changed" -> 1 aggregated diff with pages=[...]
      - 20x "Footer page N -> draft N" -> 1 aggregated diff with template "{#}"
    
    Args:
        diffs: List of diffs to process
        min_pages: Minimum number of pages to trigger aggregation (default: 2)
    
    Returns:
        List of diffs with repeating header/footer changes collapsed
    """
    keep: List[Diff] = []
    groups: Dict[Tuple[str, str, str, str], List[Diff]] = defaultdict(list)

    for d in diffs:
        md = d.metadata or {}
        hf_kind = md.get("header_footer_change")
        if not hf_kind:
            keep.append(d)
            continue

        key = (
            str(hf_kind),              # "header" | "footer"
            str(d.diff_type),          # usually "modified"
            _canonical_text_for_grouping(d.old_text),
            _canonical_text_for_grouping(d.new_text),
        )
        groups[key].append(d)

    aggregated: List[Diff] = []

    for (hf_kind, diff_type, old_key, new_key), items in groups.items():
        if len(items) < min_pages:
            keep.extend(items)
            continue

        items.sort(key=lambda x: x.page_num)
        rep = items[0]
        pages = [x.page_num for x in items]

        # Build nice templates for display
        old_tpl = _pretty_template(rep.old_text)
        new_tpl = _pretty_template(rep.new_text)

        # Merge metadata (preserve existing + add aggregation payload)
        md = dict(rep.metadata or {})
        md.update({
            "aggregated": True,
            "scope": "document",
            "pages": pages,
            "count": len(items),
            "text_template": {"old": old_tpl, "new": new_tpl},
            "examples": {
                "old_first": rep.old_text,
                "new_first": rep.new_text,
                "old_last": items[-1].old_text,
                "new_last": items[-1].new_text,
            },
        })

        conf = min(1.0, (sum(x.confidence for x in items) / len(items)) + 0.05)

        aggregated.append(Diff(
            page_num=rep.page_num,          # anchor to first occurrence
            diff_type=rep.diff_type,
            change_type=rep.change_type,    # stays "formatting" (as in detector)
            old_text=old_tpl or rep.old_text,
            new_text=new_tpl or rep.new_text,
            bbox=rep.bbox,                  # bbox of first page; pages list is in metadata
            confidence=conf,
            metadata=md,
        ))

    logger.debug(
        "Header/footer aggregation: %d diffs -> %d (aggregated %d groups)",
        len(diffs), len(keep) + len(aggregated), len(aggregated)
    )
    return keep + aggregated


# ============================================================================
# Layout Shift Aggregation
# ============================================================================

def _layout_shift_signature(d: Diff) -> Tuple[str, str, str]:
    """
    Create a grouping signature for layout shift diffs.
    
    Groups by: (change_type, subtype, shift_direction)
    Shift direction is derived from dx/dy signs to group similar shifts.
    """
    md = d.metadata or {}
    shift_info = md.get("layout_shift") or {}
    
    # Determine shift direction bucket
    dx = shift_info.get("dx", 0.0)
    dy = shift_info.get("dy", 0.0)
    
    # Bucket direction: "right"/"left"/"none" for x, "down"/"up"/"none" for y
    if abs(dx) < 1.0:
        x_dir = "stable"
    elif dx > 0:
        x_dir = "right"
    else:
        x_dir = "left"
    
    if abs(dy) < 1.0:
        y_dir = "stable"
    elif dy > 0:
        y_dir = "down"
    else:
        y_dir = "up"
    
    direction = f"{x_dir}_{y_dir}"
    subtype = md.get("subtype", "layout_shift")
    
    return (str(d.change_type), subtype, direction)


def aggregate_repeating_layout_shift_diffs(
    diffs: List[Diff],
    *,
    min_pages: int = 3,
) -> List[Diff]:
    """
    Collapse per-page layout shift diffs into document-level events.
    
    Layout shifts that repeat across multiple pages (e.g., due to a margin change)
    are aggregated into a single diff with pages list in metadata.
    
    Args:
        diffs: List of diffs to process
        min_pages: Minimum pages to trigger aggregation (default: 3)
    
    Returns:
        List with repeating layout shifts collapsed
    """
    keep: List[Diff] = []
    groups: Dict[Tuple[str, str, str], List[Diff]] = defaultdict(list)
    
    for d in diffs:
        md = d.metadata or {}
        
        # Only aggregate layout shift diffs
        is_layout_shift = (
            md.get("subtype") == "layout_shift" or
            md.get("layout_shift") is not None or
            (d.change_type == "layout" and md.get("formatting_type") in ("spacing", "page_size"))
        )
        
        if not is_layout_shift:
            keep.append(d)
            continue
        
        sig = _layout_shift_signature(d)
        groups[sig].append(d)
    
    aggregated: List[Diff] = []
    
    for sig, items in groups.items():
        if len(items) < min_pages:
            keep.extend(items)
            continue
        
        items.sort(key=lambda x: x.page_num)
        rep = items[0]
        pages = [x.page_num for x in items]
        
        # Compute average shift values
        shift_infos = [x.metadata.get("layout_shift") or {} for x in items if x.metadata]
        avg_dx = sum(s.get("dx", 0.0) for s in shift_infos) / len(shift_infos) if shift_infos else 0.0
        avg_dy = sum(s.get("dy", 0.0) for s in shift_infos) / len(shift_infos) if shift_infos else 0.0
        
        # Merge metadata
        md = dict(rep.metadata or {})
        md.update({
            "aggregated": True,
            "scope": "document",
            "pages": pages,
            "count": len(items),
            "shift_summary": {
                "avg_dx": round(avg_dx, 2),
                "avg_dy": round(avg_dy, 2),
                "direction": sig[2],  # direction from signature
            },
        })
        
        conf = min(1.0, (sum(x.confidence for x in items) / len(items)) + 0.05)
        
        aggregated.append(Diff(
            page_num=rep.page_num,
            diff_type=rep.diff_type,
            change_type=rep.change_type,
            old_text=rep.old_text,
            new_text=rep.new_text,
            bbox=rep.bbox,
            confidence=conf,
            metadata=md,
        ))
    
    logger.debug(
        "Layout shift aggregation: %d diffs -> %d (aggregated %d groups)",
        len(diffs), len(keep) + len(aggregated), len(aggregated)
    )
    return keep + aggregated


# ============================================================================
# Formatting Type Aggregation (font changes, style changes across pages)
# ============================================================================

def aggregate_repeating_formatting_diffs(
    diffs: List[Diff],
    *,
    min_pages: int = 3,
) -> List[Diff]:
    """
    Collapse repeating formatting diffs (font, size, style changes) across pages.
    
    When the same formatting change appears on multiple pages, aggregate into one.
    
    Args:
        diffs: List of diffs to process
        min_pages: Minimum pages to trigger aggregation (default: 3)
    
    Returns:
        List with repeating formatting changes collapsed
    """
    keep: List[Diff] = []
    groups: Dict[Tuple[str, str, str, str], List[Diff]] = defaultdict(list)
    
    for d in diffs:
        md = d.metadata or {}
        fmt_type = md.get("formatting_type")
        
        # Only aggregate formatting diffs with known type
        if d.change_type != "formatting" or not fmt_type:
            keep.append(d)
            continue
        
        # Skip header/footer (handled separately)
        if md.get("header_footer_change"):
            keep.append(d)
            continue
        
        # Create grouping key based on formatting type and values
        if fmt_type == "font":
            old_val = str(md.get("old_font_normalized") or md.get("old_font", ""))
            new_val = str(md.get("new_font_normalized") or md.get("new_font", ""))
        elif fmt_type == "font_size":
            old_val = str(md.get("old_size_bucket") or md.get("old_size", ""))
            new_val = str(md.get("new_size_bucket") or md.get("new_size", ""))
        elif fmt_type == "style":
            old_val = f"b{md.get('old_bold', '')}_i{md.get('old_italic', '')}"
            new_val = f"b{md.get('new_bold', '')}_i{md.get('new_italic', '')}"
        elif fmt_type == "color":
            old_val = str(md.get("old_color", ""))
            new_val = str(md.get("new_color", ""))
        else:
            keep.append(d)
            continue
        
        key = (fmt_type, str(d.diff_type), old_val, new_val)
        groups[key].append(d)
    
    aggregated: List[Diff] = []
    
    for (fmt_type, diff_type, old_val, new_val), items in groups.items():
        if len(items) < min_pages:
            keep.extend(items)
            continue
        
        items.sort(key=lambda x: x.page_num)
        rep = items[0]
        pages = sorted(set(x.page_num for x in items))
        
        md = dict(rep.metadata or {})
        md.update({
            "aggregated": True,
            "scope": "document",
            "pages": pages,
            "count": len(items),
        })
        
        conf = min(1.0, (sum(x.confidence for x in items) / len(items)) + 0.05)
        
        aggregated.append(Diff(
            page_num=rep.page_num,
            diff_type=rep.diff_type,
            change_type=rep.change_type,
            old_text=rep.old_text,
            new_text=rep.new_text,
            bbox=rep.bbox,
            confidence=conf,
            metadata=md,
        ))
    
    logger.debug(
        "Formatting aggregation: %d diffs -> %d (aggregated %d groups)",
        len(diffs), len(keep) + len(aggregated), len(aggregated)
    )
    return keep + aggregated


def fuse_diffs(
    diff_lists: List[Tuple[str, List[Diff]]],
    strategy: Literal["triangulation", "union", "intersection"] = "triangulation",
    iou_threshold: float = 0.3,
) -> List[Diff]:
    """
    Fuse diffs from multiple comparison modules with deduplication and confidence logic.
    
    Args:
        diff_lists: List of (module_name, diffs) tuples from different modules
        strategy: Fusion strategy
            - "triangulation": Merge overlapping diffs, boost confidence for multi-module agreement
            - "union": Keep all diffs, deduplicate exact overlaps only
            - "intersection": Only keep diffs detected by 2+ modules
        iou_threshold: Minimum IoU to consider diffs overlapping
    
    Returns:
        Merged list of diffs with deduplicated regions and updated confidence
    """
    if not diff_lists:
        return []
    
    total_input_diffs = sum(len(diffs) for _, diffs in diff_lists)
    total_modules = len(diff_lists)
    
    logger.info("Fusing diffs from %d modules (%d total diffs), strategy=%s",
               total_modules, total_input_diffs, strategy)
    
    # Cluster overlapping diffs
    clusters = cluster_diffs(diff_lists, iou_threshold)
    
    logger.debug("Created %d clusters from %d diffs", len(clusters), total_input_diffs)
    
    # Apply strategy
    fused_diffs: List[Diff] = []
    
    for cluster in clusters:
        num_modules = len(cluster.modules)
        
        # Apply intersection filter if needed
        if strategy == "intersection" and num_modules < 2:
            continue
        
        # Merge cluster into single diff
        old_text, new_text = merge_text_content(cluster.diffs)
        
        fused_diff = Diff(
            page_num=cluster.page_num,
            diff_type=determine_consensus_diff_type(cluster.diffs),
            change_type=determine_consensus_change_type(cluster.diffs),
            old_text=old_text,
            new_text=new_text,
            bbox=cluster.merged_bbox,
            bbox_b=cluster.merged_bbox_b,
            confidence=calculate_fused_confidence(cluster, total_modules),
            metadata=merge_metadata(cluster.diffs, cluster.modules),
        )
        
        fused_diffs.append(fused_diff)
    
    # =========================================================================
    # Document-level aggregation: collapse repeating per-page diffs
    # Order matters: header/footer first, then formatting, then layout shifts
    # =========================================================================
    
    # 1. Collapse repeating header/footer diffs (document-level events)
    fused_diffs = aggregate_repeating_header_footer_diffs(fused_diffs, min_pages=2)
    
    # 2. Collapse repeating formatting diffs (font/size/style changes)
    fused_diffs = aggregate_repeating_formatting_diffs(fused_diffs, min_pages=3)
    
    # 3. Collapse repeating layout shift diffs
    fused_diffs = aggregate_repeating_layout_shift_diffs(fused_diffs, min_pages=3)
    
    # Sort by page number and position
    fused_diffs.sort(key=lambda d: (
        d.page_num,
        d.bbox.get("y", 0.0) if d.bbox else 0.0,
        d.bbox.get("x", 0.0) if d.bbox else 0.0,
    ))
    
    logger.info("Fusion complete: %d input diffs -> %d fused diffs (%.1f%% reduction)",
               total_input_diffs,
               len(fused_diffs),
               (1 - len(fused_diffs) / max(1, total_input_diffs)) * 100)
    
    return fused_diffs


def fuse_diff_lists(*args: List[Diff], module_names: Optional[List[str]] = None) -> List[Diff]:
    """
    Convenience wrapper to fuse multiple diff lists.
    
    Args:
        *args: Variable number of diff lists
        module_names: Optional names for each module (defaults to "module_0", "module_1", etc.)
    
    Returns:
        Fused list of diffs
    
    Example:
        fused = fuse_diff_lists(text_diffs, formatting_diffs, table_diffs)
    """
    if module_names is None:
        module_names = [f"module_{i}" for i in range(len(args))]
    
    if len(module_names) != len(args):
        module_names = [f"module_{i}" for i in range(len(args))]
    
    diff_lists = list(zip(module_names, args))
    return fuse_diffs(diff_lists)

