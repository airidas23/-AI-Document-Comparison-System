"""Line-level alignment utilities."""
from __future__ import annotations

from typing import List, Optional, Tuple

from rapidfuzz import fuzz

from comparison.diff_fusion import calculate_iou
from comparison.models import Line
from utils.text_normalization import normalize_text


def _line_center_y(line: Line) -> float:
    return line.bbox["y"] + line.bbox["height"] / 2


def _line_height(line: Line) -> float:
    return line.bbox.get("height", 0.0)


def _y_distance_score(line_a: Line, line_b: Line) -> float:
    center_a = _line_center_y(line_a)
    center_b = _line_center_y(line_b)
    avg_height = (_line_height(line_a) + _line_height(line_b)) / 2
    max_distance = max(avg_height * 3.0, 1.0)
    distance = abs(center_a - center_b)
    return max(0.0, 1.0 - min(distance / max_distance, 1.0))


def _text_similarity(text_a: str, text_b: str, *, ocr: bool = False) -> float:
    if not text_a and not text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0
    norm_a = normalize_text(text_a, ocr=ocr)
    norm_b = normalize_text(text_b, ocr=ocr)
    return fuzz.token_sort_ratio(norm_a, norm_b) / 100.0


def align_lines(
    base_lines: List[Line],
    mod_lines: List[Line],
    *,
    threshold: float = 0.55,
    min_text_similarity: float = 0.0,
    is_ocr_page: bool = False,
) -> List[Tuple[Optional[Line], Optional[Line], float, str]]:
    """
    Align lines using canonical line_id, with fuzzy fallback.

    Returns tuples of (base_line, mod_line, score, reason).
    """
    if not base_lines and not mod_lines:
        return []

    base_sorted = sorted(base_lines, key=lambda line: line.reading_order)
    mod_sorted = sorted(mod_lines, key=lambda line: line.reading_order)

    mod_by_id: dict[str, List[int]] = {}
    for idx, line in enumerate(mod_sorted):
        mod_by_id.setdefault(line.line_id, []).append(idx)

    matched_base: dict[int, Tuple[int, float, str]] = {}
    matched_mod: set[int] = set()

    # Exact line_id matches.
    for idx, line in enumerate(base_sorted):
        candidates = mod_by_id.get(line.line_id, [])
        match_idx = next((c for c in candidates if c not in matched_mod), None)
        if match_idx is None:
            continue
        matched_base[idx] = (match_idx, 1.0, "line_id")
        matched_mod.add(match_idx)

    # Fuzzy matching for remaining lines.
    for idx, line in enumerate(base_sorted):
        if idx in matched_base:
            continue
        best_idx = None
        best_score = 0.0
        best_reason = "fuzzy"

        for mod_idx, mod_line in enumerate(mod_sorted):
            if mod_idx in matched_mod:
                continue
            text_score = _text_similarity(line.text, mod_line.text, ocr=is_ocr_page)
            if text_score < min_text_similarity:
                continue
            geometry_score = _y_distance_score(line, mod_line)
            iou_score = calculate_iou(line.bbox, mod_line.bbox)
            score = (0.7 * text_score) + (0.2 * geometry_score) + (0.1 * iou_score)
            if score > best_score:
                best_score = score
                best_idx = mod_idx

        if best_idx is not None and best_score >= threshold:
            matched_base[idx] = (best_idx, best_score, best_reason)
            matched_mod.add(best_idx)

    aligned: List[Tuple[Optional[Line], Optional[Line], float, str]] = []

    for idx, line in enumerate(base_sorted):
        if idx in matched_base:
            mod_idx, score, reason = matched_base[idx]
            aligned.append((line, mod_sorted[mod_idx], score, reason))
        else:
            aligned.append((line, None, 0.0, "deleted"))

    for idx, line in enumerate(mod_sorted):
        if idx not in matched_mod:
            aligned.append((None, line, 0.0, "added"))

    return aligned
