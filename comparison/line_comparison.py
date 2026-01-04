"""Line-level comparison and diff generation."""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from comparison.alignment import align_pages
from comparison.line_alignment import align_lines
from comparison.models import Diff, Line, PageData
from config.settings import settings
from utils.coordinates import normalize_bbox
from utils.text_normalization import normalize_text, compute_ocr_change_significance
from utils.diff_projection import get_word_diff_detail, bbox_union_dict


# ---------------------------------------------------------------------------
# Header/Footer band filtering (to avoid double-counting with header_footer_detector)
# ---------------------------------------------------------------------------
_PAGE_NUM_RE = re.compile(r"^(page\s*)?\d+(\s*/\s*\d+)?\s*$", re.IGNORECASE)
_RUNNING_LABEL_RE = re.compile(
    r"^(generated pair|prototype draft|synthetic pdf)\s*[—–-]\s*page\s*\d+\s*$",
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _is_in_header_footer_band(line: Line, page: PageData) -> bool:
    """
    True if line bbox overlaps with configured header/footer bands.
    
    Uses absolute coordinates (PDF points). A line is considered in header/footer
    if its top edge (y0) is within the header band, or its bottom edge (y1) is
    within the footer band.
    """
    y0 = float(line.bbox.get("y", 0.0))
    y1 = y0 + float(line.bbox.get("height", 0.0))

    header_ratio = float(getattr(settings, "header_region_height_ratio", 0.10))
    footer_ratio = float(getattr(settings, "footer_region_height_ratio", 0.10))
    
    header_y = float(page.height) * header_ratio
    footer_y = float(page.height) * (1.0 - footer_ratio)

    # Line is in header if its TOP is within header band (more aggressive)
    in_header = y0 < header_y
    # Line is in footer if its BOTTOM is within footer band (more aggressive)
    in_footer = y1 > footer_y
    
    return in_header or in_footer


def _is_header_footer_noise(line: Line, page: PageData) -> bool:
    """
    Return True if line should be excluded from line comparison.
    
    Lines in header/footer bands are handled by header_footer_detector module,
    so we exclude them here to avoid double-counting.
    """
    # Check if line is in header/footer geometric band
    if not _is_in_header_footer_band(line, page):
        return False
    
    # Line is in header/footer band - exclude it from line comparison
    # Header/footer detector will handle these separately
    return True


def _punctuation_only_change(text_a: str, text_b: str) -> bool:
    """Check if the only difference between texts is punctuation."""
    a_clean = " ".join(_PUNCT_RE.sub("", text_a).split()).casefold()
    b_clean = " ".join(_PUNCT_RE.sub("", text_b).split()).casefold()
    return (a_clean == b_clean) and (text_a.strip() != text_b.strip())


def line_changed(text_a: str, text_b: str, *, is_ocr_page: bool = False) -> Tuple[bool, Optional[dict]]:
    """
    Check if normalized line text differs, with OCR significance filtering.
    
    Returns:
        Tuple of (changed: bool, significance_info: Optional[dict])
        For OCR pages, significance_info contains change metrics.
        For non-OCR pages, detects punctuation-only changes.
    """
    norm_a = normalize_text(text_a, ocr=is_ocr_page)
    norm_b = normalize_text(text_b, ocr=is_ocr_page)
    
    if norm_a == norm_b:
        # Normalized texts are equal - check for punctuation-only changes (non-OCR only)
        if (not is_ocr_page) and _punctuation_only_change(text_a, text_b):
            return True, {"punctuation_only": True}
        return False, None
    
    # For OCR pages, check if change is significant enough to report
    if is_ocr_page:
        significance = compute_ocr_change_significance(text_a, text_b, ocr=True)
        if not significance["is_significant"]:
            # Change is too small (OCR noise) - don't report it
            return False, significance
        return True, significance
    
    return True, None


def _bbox_contains_point(bbox: dict, x: float, y: float) -> bool:
    return (
        x >= bbox.get("x", 0.0)
        and y >= bbox.get("y", 0.0)
        and x <= bbox.get("x", 0.0) + bbox.get("width", 0.0)
        and y <= bbox.get("y", 0.0) + bbox.get("height", 0.0)
    )


def _parse_region_bbox(region: dict) -> Optional[dict]:
    """Parse a layout region bbox into dict {x,y,width,height} in absolute page coords."""
    if not isinstance(region, dict):
        return None
    bbox = region.get("bbox")
    if bbox is None:
        return None

    if isinstance(bbox, dict) and {"x", "y", "width", "height"}.issubset(bbox.keys()):
        return {
            "x": float(bbox["x"]),
            "y": float(bbox["y"]),
            "width": float(bbox["width"]),
            "height": float(bbox["height"]),
        }
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}

    return None


def _line_in_table_region(line: Line, page: PageData) -> bool:
    tables = (page.metadata or {}).get("tables") or []
    if not tables:
        return False
    cx = line.bbox.get("x", 0.0) + line.bbox.get("width", 0.0) / 2
    cy = line.bbox.get("y", 0.0) + line.bbox.get("height", 0.0) / 2

    for region in tables:
        rb = _parse_region_bbox(region)
        if rb and _bbox_contains_point(rb, cx, cy):
            return True
    return False


def _line_layout_shift(
    line_a: Line,
    line_b: Line,
    page_width: float,
    page_height: float,
    *,
    tolerance_ratio: float = 0.01,
    translation: Optional[dict] = None,
) -> Optional[dict]:
    bbox_a = line_a.bbox
    bbox_b = line_b.bbox

    tx = float((translation or {}).get("dx", 0.0))
    ty = float((translation or {}).get("dy", 0.0))

    dx = abs((bbox_b["x"] - tx) - bbox_a["x"])
    dy = abs((bbox_b["y"] - ty) - bbox_a["y"])
    dw = abs(bbox_b["width"] - bbox_a["width"])
    dh = abs(bbox_b["height"] - bbox_a["height"])

    rel_dx = dx / page_width if page_width > 0 else 0.0
    rel_dy = dy / page_height if page_height > 0 else 0.0
    rel_dw = dw / page_width if page_width > 0 else 0.0
    rel_dh = dh / page_height if page_height > 0 else 0.0

    if (
        rel_dx > tolerance_ratio
        or rel_dy > tolerance_ratio
        or rel_dw > tolerance_ratio
        or rel_dh > tolerance_ratio
    ):
        return {
            "dx": dx,
            "dy": dy,
            "dw": dw,
            "dh": dh,
            "rel_dx": rel_dx,
            "rel_dy": rel_dy,
            "rel_dw": rel_dw,
            "rel_dh": rel_dh,
            "shift_detected": True,
        }
    return None


def _normalize_line_bbox(line: Line, page_width: float, page_height: float) -> dict:
    bbox = line.bbox
    return normalize_bbox(
        (bbox["x"], bbox["y"], bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]),
        page_width,
        page_height,
    )


def _compute_word_level_highlight(
    line_a: Line,
    line_b: Line,
    page_a: PageData,
    page_b: PageData,
    *,
    prefer_minimal_span: bool = False,
    allow_punctuation_tokens: bool = True,
) -> dict:
    """
    Compute word-level highlight bboxes for a line text change.
    
    Uses token data from Line.tokens for word-level bboxes.
    
    Returns dict with:
    - word_ops: list of {tag, old_tokens, new_tokens}
    - word_bboxes_a: list of normalized word bboxes in doc A
    - word_bboxes_b: list of normalized word bboxes in doc B
    - highlight_mode: "word" or "line_fallback"
    """
    # NOTE: Using difflib.SequenceMatcher here because it supports list/sequence matching.
    # rapidfuzz only supports string-to-string comparison, not list-of-tokens alignment.
    # This is acceptable since word-level highlighting is not a hot path.
    from difflib import SequenceMatcher
    
    tokens_a = line_a.tokens or []
    tokens_b = line_b.tokens or []
    
    if not tokens_a or not tokens_b:
        return {"highlight_mode": "line_fallback"}
    
    # Extract text from tokens
    texts_a = [t.text for t in tokens_a]
    texts_b = [t.text for t in tokens_b]
    
    # Normalize for matching
    norm_a = [t.text.lower().strip() for t in tokens_a]
    norm_b = [t.text.lower().strip() for t in tokens_b]
    
    sm = SequenceMatcher(a=norm_a, b=norm_b)
    
    word_ops = []
    bboxes_a = []
    bboxes_b = []
    
    opcodes = [op for op in sm.get_opcodes() if op[0] != "equal"]
    if prefer_minimal_span and opcodes:
        # For tiny OCR edits (e.g., a single-letter typo), token-level SequenceMatcher
        # can produce a large replace span. We'll still build word_ops from token
        # opcodes, but will compute highlight bboxes from a minimal *character-level*
        # diff span to keep the highlight tight.
        opcodes = sorted(opcodes, key=lambda op: max(op[2] - op[1], op[4] - op[3]))[:1]

    for tag, i1, i2, j1, j2 in opcodes:
        
        op = {
            "tag": tag,
            "a_tokens": texts_a[i1:i2],
            "b_tokens": texts_b[j1:j2],
        }
        word_ops.append(op)
        
        # Collect old side bboxes
        if tag in ("replace", "delete"):
            for k in range(i1, i2):
                if k < len(tokens_a):
                    if not allow_punctuation_tokens:
                        tx = (texts_a[k] or "").strip()
                        if tx and not any(ch.isalnum() for ch in tx):
                            continue
                    bbox = tokens_a[k].bbox
                    x0 = bbox["x"]
                    y0 = bbox["y"]
                    x1 = bbox["x"] + bbox["width"]
                    y1 = bbox["y"] + bbox["height"]
                    bboxes_a.append(normalize_bbox((x0, y0, x1, y1), page_a.width, page_a.height))
        
        # Collect new side bboxes
        if tag in ("replace", "insert"):
            for k in range(j1, j2):
                if k < len(tokens_b):
                    if not allow_punctuation_tokens:
                        tx = (texts_b[k] or "").strip()
                        if tx and not any(ch.isalnum() for ch in tx):
                            continue
                    bbox = tokens_b[k].bbox
                    x0 = bbox["x"]
                    y0 = bbox["y"]
                    x1 = bbox["x"] + bbox["width"]
                    y1 = bbox["y"] + bbox["height"]
                    bboxes_b.append(normalize_bbox((x0, y0, x1, y1), page_b.width, page_b.height))

    # If this is a tiny OCR edit, compute bboxes using the smallest character-level
    # diff span and map it back to token indices.
    if prefer_minimal_span:
        def _token_offsets(texts: List[str]) -> List[tuple[int, int]]:
            offsets: List[tuple[int, int]] = []
            pos = 0
            for t in texts:
                start = pos
                end = start + len(t)
                offsets.append((start, end))
                pos = end + 1  # space
            return offsets

        joined_a = " ".join(norm_a)
        joined_b = " ".join(norm_b)
        sm_chars = SequenceMatcher(a=joined_a, b=joined_b)
        char_ops = [op for op in sm_chars.get_opcodes() if op[0] != "equal"]

        if char_ops:
            # Pick the smallest changed span.
            tag, ai1, ai2, bj1, bj2 = min(
                char_ops,
                key=lambda op: max(op[2] - op[1], op[4] - op[3]),
            )

            off_a = _token_offsets(norm_a)
            off_b = _token_offsets(norm_b)

            def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
                return not (a1 <= b0 or b1 <= a0)

            idx_a = [i for i, (s, e) in enumerate(off_a) if _overlaps(s, e, ai1, ai2)]
            idx_b = [i for i, (s, e) in enumerate(off_b) if _overlaps(s, e, bj1, bj2)]

            if idx_a or idx_b:
                bboxes_a = []
                bboxes_b = []
                for i in idx_a:
                    if 0 <= i < len(tokens_a):
                        if not allow_punctuation_tokens:
                            tx = (texts_a[i] or "").strip()
                            if tx and not any(ch.isalnum() for ch in tx):
                                continue
                        bbox = tokens_a[i].bbox
                        x0 = bbox["x"]
                        y0 = bbox["y"]
                        x1 = bbox["x"] + bbox["width"]
                        y1 = bbox["y"] + bbox["height"]
                        bboxes_a.append(normalize_bbox((x0, y0, x1, y1), page_a.width, page_a.height))
                for i in idx_b:
                    if 0 <= i < len(tokens_b):
                        if not allow_punctuation_tokens:
                            tx = (texts_b[i] or "").strip()
                            if tx and not any(ch.isalnum() for ch in tx):
                                continue
                        bbox = tokens_b[i].bbox
                        x0 = bbox["x"]
                        y0 = bbox["y"]
                        x1 = bbox["x"] + bbox["width"]
                        y1 = bbox["y"] + bbox["height"]
                        bboxes_b.append(normalize_bbox((x0, y0, x1, y1), page_b.width, page_b.height))
    
    if not word_ops:
        return {"highlight_mode": "line_fallback"}
    
    return {
        "word_ops": word_ops,
        "word_bboxes_a": bboxes_a,
        "word_bboxes_b": bboxes_b,
        "word_bboxes": bboxes_a,  # Convenience alias for default (old doc) highlighting
        "highlight_mode": "word" if (bboxes_a or bboxes_b) else "line_fallback",
    }


def _merge_lines_to_paragraphs(lines: List[Line], *, page: Optional[PageData] = None) -> List[Line]:
    """
    Merge consecutive lines into paragraphs for OCR comparison.
    
    This reduces false positives from line-break differences in OCR output.
    Lines are merged based on vertical gap (paragraph break detection).
    
    Args:
        lines: List of Line objects from OCR
        page: Optional PageData (used to avoid merging across table regions)
    
    Returns:
        List of merged Line objects (paragraphs)
    """
    if not lines:
        return []
    
    if len(lines) == 1:
        return lines
    
    # Sort by reading order first, then y-position
    sorted_lines = sorted(lines, key=lambda l: (l.reading_order, l.bbox.get("y", 0)))
    
    def _in_table(ln: Line) -> bool:
        return bool(page) and _line_in_table_region(ln, page)  # type: ignore[arg-type]

    paragraphs: List[Line] = []
    current_para_lines: List[Line] = [sorted_lines[0]]
    current_table_flag = _in_table(sorted_lines[0])
    
    gap_threshold = settings.ocr_paragraph_gap_threshold
    
    for i in range(1, len(sorted_lines)):
        prev_line = sorted_lines[i - 1]
        curr_line = sorted_lines[i]

        # Never merge across table/non-table boundaries.
        # This prevents OCR table text from being absorbed into surrounding paragraphs,
        # which otherwise pollutes content diffs (e.g., a typo diff including table lines).
        curr_table_flag = _in_table(curr_line)
        if curr_table_flag != current_table_flag:
            para = _merge_line_group(current_para_lines)
            paragraphs.append(para)
            current_para_lines = [curr_line]
            current_table_flag = curr_table_flag
            continue
        
        prev_bottom = prev_line.bbox.get("y", 0) + prev_line.bbox.get("height", 0)
        curr_top = curr_line.bbox.get("y", 0)
        prev_height = prev_line.bbox.get("height", 12)  # default line height
        
        gap = curr_top - prev_bottom
        
        # If gap is larger than threshold * line_height, start new paragraph
        if gap > gap_threshold * prev_height:
            # Merge current paragraph lines
            para = _merge_line_group(current_para_lines)
            paragraphs.append(para)
            current_para_lines = [curr_line]
        else:
            current_para_lines.append(curr_line)
    
    # Merge remaining lines
    if current_para_lines:
        para = _merge_line_group(current_para_lines)
        paragraphs.append(para)
    
    return paragraphs


def _merge_line_group(lines: List[Line]) -> Line:
    """
    Merge a group of lines into a single paragraph Line.
    
    Args:
        lines: List of consecutive lines to merge
    
    Returns:
        Single Line representing the paragraph
    """
    if len(lines) == 1:
        return lines[0]
    
    # Combine text with space (not newline, to avoid matching issues)
    combined_text = " ".join(line.text.strip() for line in lines if line.text.strip())
    
    # Compute union bbox
    xs = [l.bbox.get("x", 0) for l in lines]
    ys = [l.bbox.get("y", 0) for l in lines]
    x2s = [l.bbox.get("x", 0) + l.bbox.get("width", 0) for l in lines]
    y2s = [l.bbox.get("y", 0) + l.bbox.get("height", 0) for l in lines]
    
    union_bbox = {
        "x": min(xs),
        "y": min(ys),
        "width": max(x2s) - min(xs),
        "height": max(y2s) - min(ys),
    }
    
    # Combine tokens from all lines
    all_tokens = []
    for line in lines:
        all_tokens.extend(line.tokens or [])
    
    # Average confidence
    confidences = [l.confidence for l in lines if l.confidence > 0]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0
    
    return Line(
        line_id=f"para_{lines[0].line_id}",
        bbox=union_bbox,
        text=combined_text,
        confidence=avg_confidence,
        reading_order=lines[0].reading_order,
        tokens=all_tokens,
        metadata={
            "is_merged_paragraph": True,
            "source_line_count": len(lines),
            "source_line_ids": [l.line_id for l in lines],
        },
    )


def compute_line_changes(
    page_a: PageData,
    page_b: PageData,
    aligned_pairs: List[Tuple[Optional[Line], Optional[Line], float, str]],
    page_confidence: float,
    *,
    is_ocr_page: bool = False,
    translation: Optional[dict] = None,
) -> List[Diff]:
    diffs: List[Diff] = []

    for base_line, mod_line, score, reason in aligned_pairs:
        # Skip header/footer lines (handled by header_footer_detector)
        if base_line and _is_header_footer_noise(base_line, page_a):
            continue
        if mod_line and _is_header_footer_noise(mod_line, page_b):
            continue
            
        if base_line and mod_line:
            text_a = base_line.text
            text_b = mod_line.text

            # OCR tables are noisy at line granularity; prefer table comparator.
            if is_ocr_page and (_line_in_table_region(base_line, page_a) or _line_in_table_region(mod_line, page_b)):
                continue

            if not text_a and not text_b:
                continue

            # Use updated line_changed with OCR significance filtering
            changed, significance = line_changed(text_a, text_b, is_ocr_page=is_ocr_page)
            if not changed:
                continue
            
            # =================================================================
            # CRITICAL: Extra sanity check for OCR pages
            # Sometimes line_changed returns True but texts are actually identical
            # after normalization. This catches edge cases.
            # =================================================================
            norm_a = normalize_text(text_a, ocr=is_ocr_page)
            norm_b = normalize_text(text_b, ocr=is_ocr_page)
            if norm_a == norm_b and not (significance and significance.get("punctuation_only")):
                # Texts are identical after normalization - skip
                continue

            bbox = _normalize_line_bbox(base_line, page_a.width, page_a.height)
            bbox_b = _normalize_line_bbox(mod_line, page_b.width, page_b.height)
            metadata = {
                "type": "line",
                "alignment_reason": reason,
                "similarity": score,
                "line_id": base_line.line_id,
                "line_id_base": base_line.line_id,
                "line_id_mod": mod_line.line_id,
                "page_width": page_a.width,
                "page_height": page_a.height,
                "is_ocr": is_ocr_page,
            }
            
            # Handle punctuation-only changes (set subtype early)
            if significance and significance.get("punctuation_only"):
                metadata["subtype"] = "punctuation_shift"
            # Add OCR significance info if available
            elif significance:
                metadata["ocr_change_ratio"] = significance.get("change_ratio", 0.0)
                metadata["ocr_changed_chars"] = significance.get("changed_chars", 0)

            # Add word-level highlight information for precise highlighting
            # For tiny OCR edits (single-char typos), prefer the smallest opcode span.
            prefer_minimal_span = False
            if is_ocr_page and significance:
                try:
                    prefer_minimal_span = int(significance.get("changed_chars", 0)) <= 3
                except Exception:
                    prefer_minimal_span = False

            word_highlight = _compute_word_level_highlight(
                base_line,
                mod_line,
                page_a,
                page_b,
                prefer_minimal_span=prefer_minimal_span,
                allow_punctuation_tokens=bool(significance and significance.get("punctuation_only")),
            )
            if word_highlight:
                metadata.update(word_highlight)

                # If we have word-level bboxes, tighten the main diff bbox to that region.
                # This greatly improves UX for OCR paragraphs where the line bbox is huge.
                if word_highlight.get("highlight_mode") == "word":
                    bba = word_highlight.get("word_bboxes_a") or []
                    bbb = word_highlight.get("word_bboxes_b") or []
                    if bba:
                        bbox = bbox_union_dict(bba)
                    if bbb:
                        bbox_b = bbox_union_dict(bbb)

            diffs.append(
                Diff(
                    page_num=page_a.page_num,
                    diff_type="modified",
                    change_type="content",
                    old_text=text_a,
                    new_text=text_b,
                    bbox=bbox,
                    bbox_b=bbox_b,
                    confidence=max(page_confidence, score),
                    metadata=metadata,
                )
            )
            continue

        if base_line and not mod_line:
            if is_ocr_page and _line_in_table_region(base_line, page_a):
                continue
            bbox = _normalize_line_bbox(base_line, page_a.width, page_a.height)
            diffs.append(
                Diff(
                    page_num=page_a.page_num,
                    diff_type="deleted",
                    change_type="content",
                    old_text=base_line.text,
                    new_text=None,
                    bbox=bbox,
                    confidence=page_confidence,
                    metadata={
                        "type": "line",
                        "alignment_reason": reason,
                        "line_id": base_line.line_id,
                        "page_width": page_a.width,
                        "page_height": page_a.height,
                        "is_ocr": is_ocr_page,
                    },
                )
            )
            continue

        if mod_line and not base_line:
            if is_ocr_page and _line_in_table_region(mod_line, page_b):
                continue
            bbox = _normalize_line_bbox(mod_line, page_b.width, page_b.height)
            diffs.append(
                Diff(
                    page_num=page_b.page_num,
                    diff_type="added",
                    change_type="content",
                    old_text=None,
                    new_text=mod_line.text,
                    bbox=bbox,
                    confidence=page_confidence,
                    metadata={
                        "type": "line",
                        "alignment_reason": reason,
                        "line_id": mod_line.line_id,
                        "page_width": page_b.width,
                        "page_height": page_b.height,
                        "is_ocr": is_ocr_page,
                    },
                )
            )

    return diffs


def _estimate_translation_from_aligned_lines(
    aligned_pairs: List[Tuple[Optional[Line], Optional[Line], float, str]],
    *,
    min_score: float = 0.9,
) -> Optional[dict]:
    """Estimate global (dx, dy) from high-confidence aligned line pairs."""
    deltas: List[tuple[float, float]] = []

    for base_line, mod_line, score, _reason in aligned_pairs:
        if base_line is None or mod_line is None:
            continue
        if score < min_score:
            continue
        dx = (mod_line.bbox["x"] + mod_line.bbox["width"] / 2) - (
            base_line.bbox["x"] + base_line.bbox["width"] / 2
        )
        dy = (mod_line.bbox["y"] + mod_line.bbox["height"] / 2) - (
            base_line.bbox["y"] + base_line.bbox["height"] / 2
        )
        deltas.append((dx, dy))

    if len(deltas) < 3:
        return None

    dxs = sorted(d[0] for d in deltas)
    dys = sorted(d[1] for d in deltas)
    mid = len(deltas) // 2
    dx = float(dxs[mid])
    dy = float(dys[mid])

    confidence = min(1.0, len(deltas) / 10.0)
    return {"dx": dx, "dy": dy, "confidence": float(confidence)}


def _detect_font_size_change(page_a: PageData, page_b: PageData) -> bool:
    """
    Detect if there's a significant font size difference between two pages.
    
    This is important for detecting line reflow - when font size changes,
    the same text may wrap to different lines, causing false positive diffs.
    
    Strategy:
    1. Compare individual line heights between pages
    2. If ANY significant height difference detected, return True
    3. This catches partial font changes (single paragraph changed)
    
    Returns True if font size difference exceeds threshold.
    """
    lines_a = page_a.lines or []
    lines_b = page_b.lines or []
    
    if not lines_a or not lines_b:
        return False
    
    threshold = settings.line_reflow_font_size_diff_threshold
    
    # Method 1: Compare line heights directly between aligned lines
    # (assumes lines are in reading order)
    def get_line_height(line: Line) -> float:
        if line.bbox and line.bbox.get("height", 0) > 0:
            return line.bbox["height"]
        return 0.0
    
    # Check if any lines have different heights
    min_lines = min(len(lines_a), len(lines_b))
    significant_diffs = 0
    
    for i in range(min_lines):
        h_a = get_line_height(lines_a[i])
        h_b = get_line_height(lines_b[i])
        if h_a > 0 and h_b > 0:
            diff = abs(h_a - h_b)
            if diff >= threshold:
                significant_diffs += 1
    
    # If at least 2 lines have significant height differences,
    # likely a font size change occurred
    if significant_diffs >= 2:
        return True
    
    # Method 2: Check if page has notably different number of lines
    # (font size change may cause text to wrap to more/fewer lines)
    if len(lines_a) > 5 and len(lines_b) > 5:
        line_count_diff = abs(len(lines_a) - len(lines_b))
        if line_count_diff >= 3:  # Significant line count difference
            return True
    
    return False


def compare_lines(
    pages_a: List[PageData],
    pages_b: List[PageData],
    alignment_map: dict | None = None,
) -> List[Diff]:
    """Compare line-level content between two documents."""
    if alignment_map is None:
        alignment_map = align_pages(pages_a, pages_b, use_similarity=True)

    page_b_lookup = {page.page_num: page for page in pages_b}
    all_diffs: List[Diff] = []
    
    # Build header/footer exclusion map to avoid double-counting
    # (headers/footers are handled by header_footer_detector separately)
    hf_exclude = _build_header_footer_exclusion(pages_a, pages_b, alignment_map)

    for page_a in pages_a:
        if page_a.page_num not in alignment_map:
            continue
        page_b_num, confidence = alignment_map[page_a.page_num]
        page_b = page_b_lookup.get(page_b_num)
        if page_b is None:
            continue

        if not page_a.lines or not page_b.lines:
            continue

        method_a = (page_a.metadata or {}).get("line_extraction_method") or (page_a.metadata or {}).get("extraction_method", "")
        method_b = (page_b.metadata or {}).get("line_extraction_method") or (page_b.metadata or {}).get("extraction_method", "")
        is_ocr_page = ("ocr" in (method_a or "").lower()) or ("ocr" in (method_b or "").lower())
        
        # Detect if font size changed between pages (triggers line reflow)
        # This helps decide if paragraph merge should be applied to digital PDFs
        has_font_size_change = _detect_font_size_change(page_a, page_b)
        
        # For OCR pages, optionally merge lines into paragraphs first (A-phase)
        # This reduces false positives from line-break differences
        lines_a = page_a.lines or []
        lines_b = page_b.lines or []
        
        # Filter out header/footer band lines (handled by header_footer_detector)
        # This is the primary filter - uses Y-band position detection
        lines_a = [ln for ln in lines_a if not _is_header_footer_noise(ln, page_a)]
        lines_b = [ln for ln in lines_b if not _is_header_footer_noise(ln, page_b)]
        
        # Secondary filter: exclude by bbox overlap with detected header/footer items
        exclude_boxes = hf_exclude.get(page_a.page_num, [])
        if exclude_boxes:
            lines_a = [ln for ln in lines_a if not _line_overlaps_any(ln, exclude_boxes)]
            lines_b = [ln for ln in lines_b if not _line_overlaps_any(ln, exclude_boxes)]
        
        # Apply paragraph merge for OCR pages OR for digital PDFs with font size changes
        # OR always for digital PDFs if always_merge_paragraphs_for_comparison is enabled
        should_merge_paragraphs = (
            (is_ocr_page and settings.ocr_paragraph_merge_enabled) or
            (not is_ocr_page and has_font_size_change and settings.line_reflow_paragraph_merge) or
            (not is_ocr_page and settings.always_merge_paragraphs_for_comparison)
        )
        
        if should_merge_paragraphs:
            lines_a = _merge_lines_to_paragraphs(lines_a, page=page_a)
            lines_b = _merge_lines_to_paragraphs(lines_b, page=page_b)

        aligned_pairs = align_lines(
            lines_a,
            lines_b,
            min_text_similarity=settings.ocr_min_text_similarity_for_match if is_ocr_page else 0.0,
            is_ocr_page=is_ocr_page,
        )

        # Estimate a coarse page-level translation from high-confidence aligned lines.
        translation = _estimate_translation_from_aligned_lines(aligned_pairs)
        if is_ocr_page and translation:
            page_a.metadata["page_alignment_translation"] = {
                "dx": float(translation.get("dx", 0.0)),
                "dy": float(translation.get("dy", 0.0)),
                "confidence": float(translation.get("confidence", 0.0)),
                "method": "median_aligned_lines",
            }

        all_diffs.extend(
            compute_line_changes(
                page_a,
                page_b,
                aligned_pairs,
                confidence,
                is_ocr_page=is_ocr_page,
                translation=translation,
            )
        )
        
        # Detect layout drift at page level (consecutive matched lines shifted in Y)
        # Use aligned pairs instead of index-based comparison to remain stable under
        # paragraph merge / line reflow.
        drift_regions = _detect_layout_drift_from_aligned_pairs(
            aligned_pairs,
            min_lines=3,
            drift_pt=6.0,
            is_ocr_page=is_ocr_page,
        )
        for region_bbox in drift_regions:
            # Normalize the region bbox
            x0 = region_bbox["x"]
            y0 = region_bbox["y"]
            x1 = region_bbox["x"] + region_bbox["width"]
            y1 = region_bbox["y"] + region_bbox["height"]
            normalized_region = normalize_bbox((x0, y0, x1, y1), page_a.width, page_a.height)
            
            all_diffs.append(
                Diff(
                    page_num=page_a.page_num,
                    diff_type="modified",
                    change_type="layout",
                    old_text=None,
                    new_text=None,
                    bbox=normalized_region,
                    confidence=confidence,
                    metadata={
                        "type": "region",
                        "subtype": "layout_drift",
                        "highlight_mode": "region",
                        "page_width": page_a.width,
                        "page_height": page_a.height,
                        "is_ocr": is_ocr_page,
                    },
                )
            )

    return all_diffs


def _detect_layout_drift_from_aligned_pairs(
    aligned_pairs,
    *,
    min_lines: int = 3,
    drift_pt: float = 6.0,
    is_ocr_page: bool = False,
):
    """Detect layout drift using aligned (base, mod) pairs.

    The previous drift detection compared base/mod lists by index, which becomes brittle
    when we merge/reflow lines into paragraphs (segmentation differs). Using aligned pairs
    makes spacing-only / layout-only changes detectable again.

    Returns a list of absolute bbox dicts (x,y,width,height) to highlight.
    """
    import statistics
    import difflib

    from utils.text_normalization import normalize_text

    matched = []
    for base_line, mod_line, _score, _reason in aligned_pairs:
        if base_line is None or mod_line is None:
            continue
        # Only consider pairs whose text is essentially unchanged.
        a = normalize_text(base_line.text or "", ocr=is_ocr_page)
        b = normalize_text(mod_line.text or "", ocr=is_ocr_page)
        if not a and not b:
            continue
        if a != b:
            if difflib.SequenceMatcher(None, a, b).ratio() < 0.97:
                continue
        matched.append((base_line, mod_line))

    # Order by base Y so "consecutive" is spatially meaningful even when alignment
    # isn't emitted in reading order.
    matched.sort(key=lambda pair: float(pair[0].bbox.get("y", 0.0)))

    if len(matched) < 2:
        return []

    dy = []
    for base_line, mod_line in matched:
        by = base_line.bbox.get("y", 0.0) + base_line.bbox.get("height", 0.0) / 2.0
        my = mod_line.bbox.get("y", 0.0) + mod_line.bbox.get("height", 0.0) / 2.0
        dy.append(my - by)

    if not dy:
        return []

    # First, detect a *step change* in vertical offset between adjacent matched units.
    # This catches paragraph spacing drift even when we have only 2 merged paragraphs.
    boundaries = [
        i
        for i in range(len(dy) - 1)
        if abs(dy[i + 1] - dy[i]) >= drift_pt
    ]

    regions = []
    if boundaries:
        for i in boundaries:
            region_bboxes = []
            for k in range(i + 1, len(matched)):
                bb = matched[k][0].bbox
                if isinstance(bb, dict):
                    region_bboxes.append(bb)
            if region_bboxes:
                regions.append(bbox_union_dict(region_bboxes))
        return regions

    # Fallback: detect longer stretches of outliers relative to the dominant offset.
    # (Useful when drift affects only a segment of many aligned lines.)
    if len(matched) < min_lines:
        return []

    med = statistics.median(dy)
    start = None
    for i, d in enumerate(dy):
        if abs(d - med) >= drift_pt:
            if start is None:
                start = i
        else:
            if start is not None and (i - start) >= min_lines:
                region_bboxes = []
                for k in range(start, i):
                    bb = matched[k][0].bbox
                    if isinstance(bb, dict):
                        region_bboxes.append(bb)
                if region_bboxes:
                    regions.append(bbox_union_dict(region_bboxes))
            start = None

    if start is not None and (len(dy) - start) >= min_lines:
        region_bboxes = []
        for k in range(start, len(dy)):
            bb = matched[k][0].bbox
            if isinstance(bb, dict):
                region_bboxes.append(bb)
        if region_bboxes:
            regions.append(bbox_union_dict(region_bboxes))

    return regions


def _build_header_footer_exclusion(
    pages_a: List[PageData],
    pages_b: List[PageData],
    alignment_map: dict,
) -> dict[int, list[dict]]:
    """
    Build header/footer exclusion zones to avoid double-counting.
    
    Returns: page_num -> list of absolute bboxes (x,y,width,height) to exclude.
    """
    try:
        from extraction.header_footer_detector import detect_headers_footers
    except ImportError:
        return {}

    try:
        hfa = detect_headers_footers(pages_a)
        hfb = detect_headers_footers(pages_b)
    except Exception:
        return {}

    lookup_b = {p.page_num: p for p in pages_b}

    out: dict[int, list[dict]] = {}
    for pa in pages_a:
        if pa.page_num not in alignment_map:
            continue
        pb_num, _conf = alignment_map[pa.page_num]
        if pb_num not in lookup_b:
            continue

        ha, fa = hfa.get(pa.page_num, ([], []))
        hb, fb = hfb.get(pb_num, ([], []))
        boxes = []
        for item in (ha + fa + hb + fb):
            bbox = getattr(item, "bbox", None)
            if bbox and isinstance(bbox, dict):
                boxes.append(bbox)
        if boxes:
            out[pa.page_num] = boxes
    return out


def _line_overlaps_any(line: Line, boxes: list[dict], *, iou_thr: float = 0.10) -> bool:
    """
    Check if a line overlaps any of the exclusion boxes (IoU >= threshold).
    """
    lx0 = line.bbox.get("x", 0.0)
    ly0 = line.bbox.get("y", 0.0)
    lx1 = lx0 + line.bbox.get("width", 0.0)
    ly1 = ly0 + line.bbox.get("height", 0.0)
    la = max(0.0, (lx1 - lx0)) * max(0.0, (ly1 - ly0))
    if la <= 0:
        return False
    for b in boxes:
        bx0 = b.get("x", 0.0)
        by0 = b.get("y", 0.0)
        bx1 = bx0 + b.get("width", 0.0)
        by1 = by0 + b.get("height", 0.0)
        ix0 = max(lx0, bx0)
        iy0 = max(ly0, by0)
        ix1 = min(lx1, bx1)
        iy1 = min(ly1, by1)
        iw = max(0.0, ix1 - ix0)
        ih = max(0.0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0:
            continue
        ba = max(0.0, (bx1 - bx0)) * max(0.0, (by1 - by0))
        union = la + ba - inter
        if union > 0 and (inter / union) >= iou_thr:
            return True
    return False
