"""Style/layout difference detection."""
from __future__ import annotations

# NOTE: Using difflib.SequenceMatcher for token-list alignment (word-level style comparison).
# rapidfuzz only supports string comparison, not list-of-tokens matching.
# This is acceptable since formatting comparison is not the performance bottleneck.
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from comparison.alignment import align_pages, align_sections
from comparison.models import Diff, PageData
from config.settings import settings
from utils.logging import logger
from utils.text_normalization import normalize_text


def _texts_similar_enough_for_formatting(norm_a: str, norm_b: str) -> bool:
    """Return True if two normalized texts are similar enough to compare formatting.

    Formatting-only edits (font size, spacing) can change line wrapping and block
    segmentation, so requiring exact equality is too strict.
    """
    a = (norm_a or "").strip()
    b = (norm_b or "").strip()
    if not a or not b:
        return False

    # If one is contained in the other, allow as long as coverage is reasonably high.
    if a in b or b in a:
        coverage = min(len(a), len(b)) / max(1, max(len(a), len(b)))
        return coverage >= 0.7

    # Otherwise require a high similarity ratio to avoid mixing in true content edits.
    return SequenceMatcher(None, a, b).ratio() >= 0.92


def _detect_any_ocr(pages_a: List[PageData], pages_b: List[PageData]) -> bool:
    """Check if any page in either document was extracted via OCR."""
    for p in pages_a + pages_b:
        md = p.metadata or {}
        extraction_method = str(md.get("extraction_method") or "")
        line_method = str(md.get("line_extraction_method") or "")
        ocr_engine = str(md.get("ocr_engine_used") or "")
        
        if (
            "ocr" in extraction_method.lower()
            or "ocr" in line_method.lower()
            or "ocr" in ocr_engine.lower()
            or "tesseract" in ocr_engine.lower()
            or "paddle" in ocr_engine.lower()
            or "deepseek" in ocr_engine.lower()
        ):
            return True
    return False


def compare_formatting(
    pages_a: List[PageData],
    pages_b: List[PageData],
    alignment_map: dict | None = None,
) -> List[Diff]:
    """
    Compare formatting (fonts, sizes, spacing, layout) between documents.
    
    Args:
        pages_a: Pages from first document
        pages_b: Pages from second document
        alignment_map: Optional pre-computed page alignment
    
    Returns:
        List of Diff objects representing formatting changes
    """
    # Early exit: skip entire formatting comparison for OCR documents if configured
    # This is the primary gate - OCR font/size data is synthetic and unreliable
    if settings.skip_formatting_for_ocr and _detect_any_ocr(pages_a, pages_b):
        logger.info(
            "Skipping formatting comparison: OCR detected and skip_formatting_for_ocr=True"
        )
        return []
    
    logger.info("Comparing formatting (threshold=%.2f)", settings.formatting_change_threshold)
    
    if alignment_map is None:
        alignment_map = align_pages(pages_a, pages_b, use_similarity=False)
    
    all_diffs: List[Diff] = []
    page_b_lookup = {page.page_num: page for page in pages_b}
    
    for page_a in pages_a:
        if page_a.page_num not in alignment_map:
            continue
        
        page_b_num, confidence = alignment_map[page_a.page_num]
        if page_b_num not in page_b_lookup:
            continue
        
        page_b = page_b_lookup[page_b_num]
        block_alignment = align_sections(page_a, page_b)

        extraction_method_a = (page_a.metadata or {}).get("extraction_method", "")
        extraction_method_b = (page_b.metadata or {}).get("extraction_method", "")
        is_ocr_page = ("ocr" in (extraction_method_a or "").lower()) or ("ocr" in (extraction_method_b or "").lower())
        
        # Compare formatting for aligned blocks
        for idx_a, idx_b in block_alignment.items():
            if idx_a >= len(page_a.blocks) or idx_b >= len(page_b.blocks):
                continue
            
            block_a = page_a.blocks[idx_a]
            block_b = page_b.blocks[idx_b]
            
            # Skip if text content is different (handled by text comparison)
            # Use normalized comparison to ignore case and minor differences
            norm_a = normalize_text(block_a.text, ocr=is_ocr_page)
            norm_b = normalize_text(block_b.text, ocr=is_ocr_page)
            if norm_a != norm_b and not _texts_similar_enough_for_formatting(norm_a, norm_b):
                continue
            
            # Compare styles (pass page dimensions for normalization)
            style_diffs = _compare_styles(
                block_a, block_b, page_a.page_num, confidence,
                page_a.width, page_a.height,
                extraction_method_a=extraction_method_a,
                extraction_method_b=extraction_method_b,
            )
            all_diffs.extend(style_diffs)
        
        # Compare page-level layout
        layout_diffs = _compare_page_layout(page_a, page_b, confidence)
        all_diffs.extend(layout_diffs)
    
    logger.info("Detected %d formatting differences", len(all_diffs))
    return all_diffs


def _compare_styles(
    block_a, block_b, page_num: int, confidence: float,
    page_width: float, page_height: float,
    *,
    extraction_method_a: str = "",
    extraction_method_b: str = "",
) -> List[Diff]:
    """Compare style attributes between two text blocks using fingerprint-based comparison."""
    diffs: List[Diff] = []
    
    style_a = block_a.style
    style_b = block_b.style
    
    # Create default style if needed for fingerprint comparison
    from comparison.models import Style
    style_a = style_a or Style()
    style_b = style_b or Style()
    
    # Skip if both styles are empty (no formatting to compare)
    if not style_a.font and not style_a.size and not style_b.font and not style_b.size:
        return diffs
    
    # Skip formatting comparison for OCR if configured
    is_ocr_a = "ocr" in (extraction_method_a or "").lower()
    is_ocr_b = "ocr" in (extraction_method_b or "").lower()
    
    if settings.skip_formatting_for_ocr and (is_ocr_a or is_ocr_b):
        # Skip formatting comparison when OCR is used (styles are unreliable)
        logger.debug(
            "Skipping formatting comparison for OCR-extracted blocks (extraction_method: %s, %s)",
            extraction_method_a, extraction_method_b
        )
        return diffs  # Return empty diffs
    
    # Adjust confidence for OCR - OCR styles are often synthetic (if not skipping)
    adjusted_confidence = confidence
    if is_ocr_a or is_ocr_b:
        adjusted_confidence = confidence * 0.7  # Reduce confidence for OCR
    
    # Normalize bbox coordinates for all diffs
    normalized_bbox = block_a.normalize_bbox(page_width, page_height)
    
    # Use fingerprint-based comparison for deterministic results
    fp_a = style_a.get_fingerprint()
    fp_b = style_b.get_fingerprint()
    
    block_font_size_reported = False

    # Compare font family (normalized)
    if fp_a["font_family_normalized"] != fp_b["font_family_normalized"]:
        diffs.append(Diff(
            page_num=page_num,
            diff_type="modified",
            change_type="formatting",
            old_text=block_a.text,
            new_text=block_b.text,
            bbox=normalized_bbox,
            confidence=adjusted_confidence,
            metadata={
                "formatting_type": "font",
                "old_font": style_a.font,
                "new_font": style_b.font,
                "old_font_normalized": fp_a["font_family_normalized"],
                "new_font_normalized": fp_b["font_family_normalized"],
                "page_width": page_width,
                "page_height": page_height,
            },
        ))
    
    # Compare font size (using bucket comparison with noise gate)
    if style_a.size and style_b.size:
        size_a = style_a.size
        size_b = style_b.size
        size_diff = abs(size_a - size_b)
        
        # Noise gate: ignore tiny differences (< 0.5pt)
        noise_gate_threshold = 0.5
        
        # Only report if difference is meaningful
        # If buckets differ, it means normalized size differs (already accounts for bucketing)
        # But we still apply noise gate to ignore very small raw differences
        if size_diff >= noise_gate_threshold:
            # Report if buckets differ (normalized comparison) or absolute diff exceeds threshold
            if fp_a["size_bucket"] != fp_b["size_bucket"]:
                # Bucket differs - this is a real difference after normalization
                diffs.append(Diff(
                    page_num=page_num,
                    diff_type="modified",
                    change_type="formatting",
                    old_text=block_a.text,
                    new_text=block_b.text,
                    bbox=normalized_bbox,
                    confidence=adjusted_confidence,
                    metadata={
                        "formatting_type": "font_size",
                        "old_size": style_a.size,
                        "new_size": style_b.size,
                        "old_size_bucket": fp_a["size_bucket"],
                        "new_size_bucket": fp_b["size_bucket"],
                        "size_diff": size_diff,
                        "page_width": page_width,
                        "page_height": page_height,
                    },
                ))
                block_font_size_reported = True
            elif size_diff >= settings.font_size_change_threshold_pt:
                # Buckets same but absolute diff exceeds threshold (fallback for edge cases)
                diffs.append(Diff(
                    page_num=page_num,
                    diff_type="modified",
                    change_type="formatting",
                    old_text=block_a.text,
                    new_text=block_b.text,
                    bbox=normalized_bbox,
                    confidence=adjusted_confidence,
                    metadata={
                        "formatting_type": "font_size",
                        "old_size": style_a.size,
                        "new_size": style_b.size,
                        "old_size_bucket": fp_a["size_bucket"],
                        "new_size_bucket": fp_b["size_bucket"],
                        "size_diff": size_diff,
                        "page_width": page_width,
                        "page_height": page_height,
                    },
                ))
                block_font_size_reported = True
    
    # Compare bold/italic (using fingerprint)
    if fp_a["weight"] != fp_b["weight"] or fp_a["slant"] != fp_b["slant"]:
        diffs.append(Diff(
            page_num=page_num,
            diff_type="modified",
            change_type="formatting",
            old_text=block_a.text,
            new_text=block_b.text,
            bbox=normalized_bbox,
            confidence=adjusted_confidence,
            metadata={
                "formatting_type": "style",
                "old_bold": style_a.bold,
                "old_italic": style_a.italic,
                "new_bold": style_b.bold,
                "new_italic": style_b.italic,
                "old_weight": fp_a["weight"],
                "new_weight": fp_b["weight"],
                "old_slant": fp_a["slant"],
                "new_slant": fp_b["slant"],
                "page_width": page_width,
                "page_height": page_height,
            },
        ))
    
    # Compare color
    if style_a.color and style_b.color:
        color_diff = sum(abs(a - b) for a, b in zip(style_a.color, style_b.color))
        if color_diff > settings.color_difference_threshold:
            diffs.append(Diff(
                page_num=page_num,
                diff_type="modified",
                change_type="formatting",
                old_text=block_a.text,
                new_text=block_b.text,
                bbox=normalized_bbox,
                confidence=adjusted_confidence,
            metadata={
                "formatting_type": "color",
                    "old_color": style_a.color,
                    "new_color": style_b.color,
                    "page_width": page_width,
                    "page_height": page_height,
                },
            ))

    # Word-level formatting (only when word metadata is available)
    diffs.extend(
        _compare_word_styles(
            block_a,
            block_b,
            page_num,
            adjusted_confidence,
            page_width,
            page_height,
            suppress_font_size=block_font_size_reported,
        )
    )
    
    return diffs


def _normalize_bbox_dict(b: Dict[str, float], page_width: float, page_height: float) -> Dict[str, float]:
    """Normalize a bbox dict (absolute pt) to 0-1 coordinates."""
    x = float(b.get("x", 0.0))
    y = float(b.get("y", 0.0))
    w = float(b.get("width", 0.0))
    h = float(b.get("height", 0.0))
    pw = float(page_width) if page_width else 1.0
    ph = float(page_height) if page_height else 1.0
    return {
        "x": x / pw,
        "y": y / ph,
        "width": w / pw,
        "height": h / ph,
    }


def _style_from_word_meta(word: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    st = word.get("style")
    return st if isinstance(st, dict) else None


def _style_color_tuple(st: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
    c = st.get("color")
    if isinstance(c, tuple) and len(c) == 3:
        try:
            return (int(c[0]), int(c[1]), int(c[2]))
        except Exception:
            return None
    if isinstance(c, list) and len(c) == 3:
        try:
            return (int(c[0]), int(c[1]), int(c[2]))
        except Exception:
            return None
    return None


def _compare_word_styles(
    block_a,
    block_b,
    page_num: int,
    confidence: float,
    page_width: float,
    page_height: float,
    *,
    suppress_font_size: bool = False,
) -> List[Diff]:
    """Compare per-word styles when metadata['words'][].style is available."""
    diffs: List[Diff] = []

    meta_a = getattr(block_a, "metadata", None) or {}
    meta_b = getattr(block_b, "metadata", None) or {}
    words_a = meta_a.get("words") or []
    words_b = meta_b.get("words") or []
    if not isinstance(words_a, list) or not isinstance(words_b, list):
        return diffs

    # Only proceed if at least one side actually has word style info.
    if not any(isinstance(_style_from_word_meta(w), dict) for w in words_a) and not any(
        isinstance(_style_from_word_meta(w), dict) for w in words_b
    ):
        return diffs

    tokens_a = [normalize_text(str(w.get("text", ""))) for w in words_a]
    tokens_b = [normalize_text(str(w.get("text", ""))) for w in words_b]
    if not tokens_a or not tokens_b:
        return diffs

    sm = SequenceMatcher(a=tokens_a, b=tokens_b)

    # Word-level size threshold: smaller than block threshold, but still noise-gated.
    size_threshold = max(0.3, float(settings.font_size_change_threshold_pt) * 0.3)

    from utils.style_normalization import normalize_font_name

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            continue
        n = min(i2 - i1, j2 - j1)
        for k in range(n):
            wa = words_a[i1 + k]
            wb = words_b[j1 + k]
            if not isinstance(wa, dict) or not isinstance(wb, dict):
                continue
            bbox_a = wa.get("bbox")
            if not isinstance(bbox_a, dict):
                continue
            st_a = _style_from_word_meta(wa)
            st_b = _style_from_word_meta(wb)
            if not st_a or not st_b:
                continue

            word_text = str(wa.get("text", ""))

            # Font family
            fa = normalize_font_name(str(st_a.get("font") or ""))
            fb = normalize_font_name(str(st_b.get("font") or ""))
            if fa and fb and fa != fb:
                diffs.append(
                    Diff(
                        page_num=page_num,
                        diff_type="modified",
                        change_type="formatting",
                        old_text=word_text,
                        new_text=word_text,
                        bbox=_normalize_bbox_dict(bbox_a, page_width, page_height),
                        confidence=confidence,
                        metadata={
                            "formatting_type": "font",
                            "scope": "word",
                            "word_text": word_text,
                            "line_text": getattr(block_a, "text", ""),
                            "old_font": st_a.get("font"),
                            "new_font": st_b.get("font"),
                            "old_font_normalized": fa,
                            "new_font_normalized": fb,
                            "page_width": page_width,
                            "page_height": page_height,
                        },
                    )
                )

            # Font size
            sa = st_a.get("size")
            sb = st_b.get("size")
            try:
                sa_f = None if sa is None else float(sa)
                sb_f = None if sb is None else float(sb)
            except Exception:
                sa_f = None
                sb_f = None
            if (not suppress_font_size) and sa_f is not None and sb_f is not None:
                sd = abs(sa_f - sb_f)
                if sd >= size_threshold:
                    diffs.append(
                        Diff(
                            page_num=page_num,
                            diff_type="modified",
                            change_type="formatting",
                            old_text=word_text,
                            new_text=word_text,
                            bbox=_normalize_bbox_dict(bbox_a, page_width, page_height),
                            confidence=confidence,
                            metadata={
                                "formatting_type": "font_size",
                                "scope": "word",
                                "word_text": word_text,
                                "line_text": getattr(block_a, "text", ""),
                                "old_size": sa_f,
                                "new_size": sb_f,
                                "size_diff": sd,
                                "threshold_pt": size_threshold,
                                "page_width": page_width,
                                "page_height": page_height,
                            },
                        )
                    )

            # Bold/italic
            ba = bool(st_a.get("bold"))
            bb = bool(st_b.get("bold"))
            ia = bool(st_a.get("italic"))
            ib = bool(st_b.get("italic"))
            if ba != bb or ia != ib:
                diffs.append(
                    Diff(
                        page_num=page_num,
                        diff_type="modified",
                        change_type="formatting",
                        old_text=word_text,
                        new_text=word_text,
                        bbox=_normalize_bbox_dict(bbox_a, page_width, page_height),
                        confidence=confidence,
                        metadata={
                            "formatting_type": "style",
                            "scope": "word",
                            "word_text": word_text,
                            "line_text": getattr(block_a, "text", ""),
                            "old_bold": ba,
                            "old_italic": ia,
                            "new_bold": bb,
                            "new_italic": ib,
                            "page_width": page_width,
                            "page_height": page_height,
                        },
                    )
                )

            # Color
            ca = _style_color_tuple(st_a)
            cb = _style_color_tuple(st_b)
            if ca and cb:
                color_diff = sum(abs(x - y) for x, y in zip(ca, cb))
                if color_diff > settings.color_difference_threshold:
                    diffs.append(
                        Diff(
                            page_num=page_num,
                            diff_type="modified",
                            change_type="formatting",
                            old_text=word_text,
                            new_text=word_text,
                            bbox=_normalize_bbox_dict(bbox_a, page_width, page_height),
                            confidence=confidence,
                            metadata={
                                "formatting_type": "color",
                                "scope": "word",
                                "word_text": word_text,
                                "line_text": getattr(block_a, "text", ""),
                                "old_color": ca,
                                "new_color": cb,
                                "page_width": page_width,
                                "page_height": page_height,
                            },
                        )
                    )

    return diffs


def _compare_page_layout(page_a: PageData, page_b: PageData, confidence: float) -> List[Diff]:
    """Compare page-level layout differences."""
    diffs: List[Diff] = []
    
    # Compare page dimensions
    width_diff = abs(page_a.width - page_b.width) / max(page_a.width, page_b.width)
    height_diff = abs(page_a.height - page_b.height) / max(page_a.height, page_b.height)
    
    if width_diff > settings.formatting_change_threshold or height_diff > settings.formatting_change_threshold:
        diffs.append(Diff(
            page_num=page_a.page_num,
            diff_type="modified",
            change_type="layout",
            old_text=None,
            new_text=None,
            bbox=None,
            confidence=confidence,
            metadata={
                "formatting_type": "page_size",
                "old_size": (page_a.width, page_a.height),
                "new_size": (page_b.width, page_b.height),
            },
        ))
    
    # Compare spacing between blocks (simple heuristic)
    if len(page_a.blocks) > 1 and len(page_b.blocks) > 1:
        spacing_a = _calculate_average_spacing(page_a)
        spacing_b = _calculate_average_spacing(page_b)
        
        if spacing_a > 0 and spacing_b > 0:
            spacing_diff = abs(spacing_a - spacing_b) / max(spacing_a, spacing_b)
            if spacing_diff > settings.formatting_change_threshold:
                diffs.append(Diff(
                    page_num=page_a.page_num,
                    diff_type="modified",
                    change_type="layout",
                    old_text=None,
                    new_text=None,
                    bbox=None,
                    confidence=confidence,
                    metadata={
                        "formatting_type": "spacing",
                        "old_spacing": spacing_a,
                        "new_spacing": spacing_b,
                    },
                ))
    
    return diffs


def _calculate_average_spacing(page: PageData) -> float:
    """Calculate average vertical spacing between blocks."""
    if len(page.blocks) < 2:
        return 0.0
    
    spacings = []
    sorted_blocks = sorted(page.blocks, key=lambda b: (b.bbox["y"], b.bbox["x"]))
    
    for i in range(len(sorted_blocks) - 1):
        block_a = sorted_blocks[i]
        block_b = sorted_blocks[i + 1]
        
        # Vertical spacing: distance from bottom of block_a to top of block_b
        # bbox is {"x": x, "y": y, "width": w, "height": h}
        bottom_a = block_a.bbox["y"] + block_a.bbox["height"]
        top_b = block_b.bbox["y"]
        spacing = top_b - bottom_a
        if spacing > 0:
            spacings.append(spacing)
    
    return sum(spacings) / len(spacings) if spacings else 0.0
