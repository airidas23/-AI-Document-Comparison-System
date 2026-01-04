"""
Diff projection utilities for word-level and layout-level highlighting.

This module provides algorithms to project text changes onto word-level bboxes
and detect layout drift for region-level highlighting.
"""
from __future__ import annotations

import statistics
# NOTE: Using difflib.SequenceMatcher for token-list alignment (word-level diff projection).
# rapidfuzz only supports string comparison, not list-of-tokens matching with opcodes.
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

from utils.logging import logger


def word_list_from_lineblock(line_block: Any) -> Tuple[List[str], List[Dict]]:
    """
    Extract word text and word metadata from a line block.
    
    Args:
        line_block: TextBlock with metadata["words"]
    
    Returns:
        Tuple of (list of word texts, list of word dicts with bbox)
    """
    meta = getattr(line_block, "metadata", None) or {}
    words = meta.get("words") or []
    if not words:
        return [], []
    texts = [str(w.get("text", "")) for w in words]
    return texts, words


def project_text_change_to_word_bboxes(
    old_line: Any,
    new_line: Any,
) -> Tuple[List[Dict[str, float]], str]:
    """
    Project a text change onto specific word bboxes for precise highlighting.
    
    Uses token-level diff on word sequences to identify which words changed.
    
    Args:
        old_line: TextBlock from base document
        new_line: TextBlock from modified document
    
    Returns:
        Tuple of (list of word bboxes to highlight, highlight_mode)
        highlight_mode is "word" if word-level, "line_fallback" if falling back to line bbox
    """
    old_tokens, old_words = word_list_from_lineblock(old_line)
    new_tokens, new_words = word_list_from_lineblock(new_line)

    if not old_tokens or not new_tokens:
        # Fallback: return line bbox
        old_bbox = getattr(old_line, "bbox", None)
        if old_bbox:
            return [old_bbox], "line_fallback"
        return [], "no_bbox"

    sm = SequenceMatcher(a=old_tokens, b=new_tokens)
    highlight: List[Dict[str, float]] = []

    def _is_alnum_token(token_text: str) -> bool:
        return any(ch.isalnum() for ch in (token_text or ""))

    def _nearest_alnum_bbox(words: List[Dict], start_idx: int, *, direction: int) -> Dict[str, float] | None:
        i = start_idx
        while 0 <= i < len(words):
            tx = str(words[i].get("text", ""))
            if _is_alnum_token(tx) and isinstance(words[i].get("bbox"), dict):
                return words[i]["bbox"]
            i += direction
        return None

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        
        # For replace/delete: highlight old side
        # For insert: highlight closest old token (or whole line)
        if tag in ("replace", "delete"):
            for k in range(i1, i2):
                if k < len(old_words) and "bbox" in old_words[k]:
                    highlight.append(old_words[k]["bbox"])
        elif tag == "insert":
            # Insert has no old bbox; highlight nearest neighbor if possible
            cand = None
            if i1 > 0:
                cand = _nearest_alnum_bbox(old_words, i1 - 1, direction=-1)
            if cand is None:
                cand = _nearest_alnum_bbox(old_words, i1, direction=1)
            if cand is None and old_words and isinstance(old_words[0].get("bbox"), dict):
                cand = old_words[0]["bbox"]
            if cand is not None:
                highlight.append(cand)

    # De-duplicate bboxes
    uniq: List[Dict[str, float]] = []
    seen = set()
    for b in highlight:
        if not isinstance(b, dict):
            continue
        key = (b.get("x", 0), b.get("y", 0), b.get("width", 0), b.get("height", 0))
        if key not in seen:
            seen.add(key)
            uniq.append(b)

    if not uniq:
        # Fallback to line bbox
        old_bbox = getattr(old_line, "bbox", None)
        if old_bbox:
            return [old_bbox], "line_fallback"
        return [], "no_bbox"

    return uniq, "word"


def project_text_change_to_new_word_bboxes(
    old_line: Any,
    new_line: Any,
) -> Tuple[List[Dict[str, float]], str]:
    """
    Project a text change onto specific word bboxes in the NEW document.
    
    This is useful for highlighting changes on the modified document side.
    
    Args:
        old_line: TextBlock from base document
        new_line: TextBlock from modified document
    
    Returns:
        Tuple of (list of word bboxes to highlight in new doc, highlight_mode)
    """
    old_tokens, old_words = word_list_from_lineblock(old_line)
    new_tokens, new_words = word_list_from_lineblock(new_line)

    if not old_tokens or not new_tokens:
        new_bbox = getattr(new_line, "bbox", None)
        if new_bbox:
            return [new_bbox], "line_fallback"
        return [], "no_bbox"

    sm = SequenceMatcher(a=old_tokens, b=new_tokens)
    highlight: List[Dict[str, float]] = []

    def _is_alnum_token(token_text: str) -> bool:
        return any(ch.isalnum() for ch in (token_text or ""))

    def _nearest_alnum_bbox(words: List[Dict], start_idx: int, *, direction: int) -> Dict[str, float] | None:
        i = start_idx
        while 0 <= i < len(words):
            tx = str(words[i].get("text", ""))
            if _is_alnum_token(tx) and isinstance(words[i].get("bbox"), dict):
                return words[i]["bbox"]
            i += direction
        return None

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        
        # For replace/insert: highlight new side
        if tag in ("replace", "insert"):
            for k in range(j1, j2):
                if k < len(new_words) and "bbox" in new_words[k]:
                    highlight.append(new_words[k]["bbox"])
        elif tag == "delete":
            # Delete has no new bbox; highlight nearest neighbor
            cand = None
            if j1 > 0:
                cand = _nearest_alnum_bbox(new_words, j1 - 1, direction=-1)
            if cand is None:
                cand = _nearest_alnum_bbox(new_words, j1, direction=1)
            if cand is None and new_words and isinstance(new_words[0].get("bbox"), dict):
                cand = new_words[0]["bbox"]
            if cand is not None:
                highlight.append(cand)

    # De-duplicate
    uniq: List[Dict[str, float]] = []
    seen = set()
    for b in highlight:
        if not isinstance(b, dict):
            continue
        key = (b.get("x", 0), b.get("y", 0), b.get("width", 0), b.get("height", 0))
        if key not in seen:
            seen.add(key)
            uniq.append(b)

    if not uniq:
        new_bbox = getattr(new_line, "bbox", None)
        if new_bbox:
            return [new_bbox], "line_fallback"
        return [], "no_bbox"

    return uniq, "word"


def line_center_y(line_block: Any) -> float:
    """Get the vertical center of a line block."""
    b = getattr(line_block, "bbox", None)
    if b is None:
        return 0.0
    if isinstance(b, dict):
        y0 = b.get("y", 0.0)
        h = b.get("height", 0.0)
        return y0 + h / 2.0
    return 0.0


def bbox_union_dict(bboxes: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute the union of multiple bboxes."""
    if not bboxes:
        return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
    
    xs = [b.get("x", 0.0) for b in bboxes]
    ys = [b.get("y", 0.0) for b in bboxes]
    x2s = [b.get("x", 0.0) + b.get("width", 0.0) for b in bboxes]
    y2s = [b.get("y", 0.0) + b.get("height", 0.0) for b in bboxes]
    
    x0, y0 = min(xs), min(ys)
    x1, y1 = max(x2s), max(y2s)
    
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def detect_layout_drift(
    base_lines: List[Any],
    mod_lines: List[Any],
    *,
    min_lines: int = 3,
    drift_pt: float = 6.0,
) -> List[Dict[str, float]]:
    """
    Detect layout drift between base and modified line blocks.
    
    Layout drift means: text is the same but line positions shifted in Y.
    This returns region bboxes for highlighting.
    
    Args:
        base_lines: List of TextBlocks from base document
        mod_lines: List of TextBlocks from modified document
        min_lines: Minimum consecutive lines to consider as a drift region
        drift_pt: Threshold in points to consider as significant drift
    
    Returns:
        List of region bboxes (unions of consecutive drifted lines)
    """
    n = min(len(base_lines), len(mod_lines))
    if n < min_lines:
        return []

    # Calculate Y-center differences
    dy = []
    for i in range(n):
        base_y = line_center_y(base_lines[i])
        mod_y = line_center_y(mod_lines[i])
        dy.append(mod_y - base_y)

    if not dy:
        return []

    # Robust baseline: median shift
    med = statistics.median(dy)
    
    # Find segments that deviate from median by more than threshold
    regions: List[Dict[str, float]] = []
    start = None
    
    for i, d in enumerate(dy):
        if abs(d - med) >= drift_pt:
            if start is None:
                start = i
        else:
            if start is not None and (i - start) >= min_lines:
                # Extract bboxes for this region
                region_bboxes = []
                for k in range(start, i):
                    bbox = getattr(base_lines[k], "bbox", None)
                    if isinstance(bbox, dict):
                        region_bboxes.append(bbox)
                if region_bboxes:
                    regions.append(bbox_union_dict(region_bboxes))
            start = None

    # Handle trailing region
    if start is not None and (n - start) >= min_lines:
        region_bboxes = []
        for k in range(start, n):
            bbox = getattr(base_lines[k], "bbox", None)
            if isinstance(bbox, dict):
                region_bboxes.append(bbox)
        if region_bboxes:
            regions.append(bbox_union_dict(region_bboxes))

    if regions:
        logger.debug("Detected %d layout drift regions", len(regions))
    
    return regions


def get_word_diff_detail(
    old_line: Any,
    new_line: Any,
) -> Dict[str, Any]:
    """
    Get detailed word-level diff information.
    
    Returns a dict with:
    - ops: List of diff operations (replace/insert/delete)
    - old_bboxes: Bboxes to highlight in old document
    - new_bboxes: Bboxes to highlight in new document
    - highlight_mode: "word" or "line_fallback"
    """
    old_tokens, old_words = word_list_from_lineblock(old_line)
    new_tokens, new_words = word_list_from_lineblock(new_line)

    result = {
        "ops": [],
        "old_bboxes": [],
        "new_bboxes": [],
        "highlight_mode": "word",
    }

    if not old_tokens or not new_tokens:
        result["highlight_mode"] = "line_fallback"
        old_bbox = getattr(old_line, "bbox", None)
        new_bbox = getattr(new_line, "bbox", None)
        if old_bbox:
            result["old_bboxes"] = [old_bbox]
        if new_bbox:
            result["new_bboxes"] = [new_bbox]
        return result

    sm = SequenceMatcher(a=old_tokens, b=new_tokens)
    
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        
        op = {
            "tag": tag,
            "old_tokens": old_tokens[i1:i2],
            "new_tokens": new_tokens[j1:j2],
        }
        result["ops"].append(op)

        # Collect old side bboxes
        if tag in ("replace", "delete"):
            for k in range(i1, i2):
                if k < len(old_words) and "bbox" in old_words[k]:
                    result["old_bboxes"].append(old_words[k]["bbox"])

        # Collect new side bboxes
        if tag in ("replace", "insert"):
            for k in range(j1, j2):
                if k < len(new_words) and "bbox" in new_words[k]:
                    result["new_bboxes"].append(new_words[k]["bbox"])

    if not result["old_bboxes"] and not result["new_bboxes"]:
        result["highlight_mode"] = "line_fallback"

    return result
