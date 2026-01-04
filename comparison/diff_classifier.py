"""Classify diffs into content vs formatting types."""
from __future__ import annotations

import re
from typing import List

from comparison.models import ChangeType, Diff, DiffType
from utils.logging import logger
from utils.text_normalization import normalize_text
from utils.text_diff import has_character_content_difference


def _ensure_subtype(diff: Diff) -> None:
    """
    Ensure diff.metadata['subtype'] is set for non-content diffs too.
    Keeps existing subtype if already provided upstream.
    """
    if diff.metadata is None:
        diff.metadata = {}
    md = diff.metadata

    if md.get("subtype"):
        return  # don't override

    # Highest-signal, explicit detectors first
    if md.get("header_footer_change"):
        # header_footer_change is expected to be "header" or "footer"
        md["subtype"] = f"header_footer:{md['header_footer_change']}"
        return

    if md.get("table_change"):
        md["subtype"] = f"table:{md['table_change']}"
        return

    if md.get("figure_change"):
        md["subtype"] = f"figure:{md['figure_change']}"
        return

    # Formatting diffs
    if diff.change_type == "formatting":
        ft = md.get("formatting_type") or "unknown"
        scope = md.get("scope")  # "word" / "line" / "block" if set upstream
        md["subtype"] = f"format:{ft}" + (f":{scope}" if scope else "")
        return

    # Layout diffs
    if diff.change_type == "layout":
        if md.get("layout_shift"):
            md["subtype"] = "layout_shift"
        elif md.get("layout_drift_region"):
            md["subtype"] = "layout_drift"
        else:
            md["subtype"] = "layout"
        return

    # Visual diffs (images / render)
    if diff.change_type == "visual":
        md["subtype"] = "visual"
        return

    # Fallback: leave for downstream classification to fill
    # (content diffs get subtype from _classify_single_diff rules)


def classify_diffs(diffs: List[Diff]) -> List[Diff]:
    """
    Classify differences using rule-based heuristics.
    
    Categorizes diffs into subtypes:
    - content: semantic changes, paraphrasing, additions/deletions
    - formatting: font, size, color, style changes
    - layout: spacing, page size, column changes
    - visual: pixel-level differences
    
    For OCR diffs, applies additional filtering:
    - Reduces confidence for formatting/layout diffs (OCR styles are synthetic)
    - Marks microscopic changes as low-severity
    - REMOVES diffs that are likely OCR noise (punctuation, whitespace, single-char)
    
    Args:
        diffs: List of unclassified Diff objects
    
    Returns:
        List of Diff objects with updated change_type and metadata
    """
    from config.settings import settings
    
    logger.info("Classifying %d diffs", len(diffs))
    
    classified = []
    for diff in diffs:
        classified_diff = _classify_single_diff(diff)
        # Post-classification: apply OCR-specific adjustments
        classified_diff = _apply_ocr_adjustments(classified_diff)
        classified.append(classified_diff)
    
    # CRITICAL: Apply aggressive OCR noise filter AFTER classification
    # This removes diffs that are likely OCR artifacts
    if settings.ocr_aggressive_noise_filter:
        before_count = len(classified)
        classified = _filter_ocr_noise_diffs(classified)
        filtered_count = before_count - len(classified)
        if filtered_count > 0:
            logger.info("OCR noise filter: removed %d/%d diffs (%.1f%%)", 
                       filtered_count, before_count, 
                       filtered_count / max(1, before_count) * 100)
    
    logger.debug("Classification complete: %d diffs processed", len(classified))
    return classified


def _filter_ocr_noise_diffs(diffs: List[Diff]) -> List[Diff]:
    """
    Aggressively filter OCR noise diffs.
    
    Removes diffs that are likely OCR artifacts:
    - Punctuation-only changes (period/comma confusion, dash variants)
    - Whitespace-only changes (OCR spacing is synthetic)
    - Case-only changes (OCR case confusion)
    - Single-character changes below threshold
    - Formatting diffs on OCR pages (font/size is synthetic)
    - Layout diffs on OCR pages (bbox positions have natural variance)
    
    Args:
        diffs: List of classified Diff objects
    
    Returns:
        Filtered list with OCR noise removed
    """
    from config.settings import settings
    
    filtered = []
    
    for diff in diffs:
        # Only filter OCR diffs
        if not _diff_is_ocr(diff):
            filtered.append(diff)
            continue
        
        md = diff.metadata or {}
        subtype = md.get("subtype", "")
        
        # Filter punctuation-only diffs
        if settings.ocr_ignore_punctuation_diffs:
            if subtype in ("punctuation", "punctuation_shift"):
                logger.debug("Filtering OCR punctuation diff: page=%s", diff.page_num)
                continue
        
        # Filter whitespace-only diffs
        if settings.ocr_ignore_whitespace_diffs:
            if subtype == "whitespace":
                logger.debug("Filtering OCR whitespace diff: page=%s", diff.page_num)
                continue
        
        # Filter case-only diffs
        if settings.ocr_ignore_case_diffs:
            if subtype == "case":
                logger.debug("Filtering OCR case diff: page=%s", diff.page_num)
                continue
        
        # Filter formatting diffs (OCR font/size is synthetic)
        if diff.change_type == "formatting":
            # Keep only significant formatting changes (e.g., bold->normal is detectable)
            # Filter spacing/font-size which are entirely synthetic
            formatting_type = md.get("formatting_type", "")
            if formatting_type in ("spacing", "font_size", "font", "line_spacing"):
                logger.debug("Filtering OCR formatting diff: page=%s type=%s", 
                           diff.page_num, formatting_type)
                continue
        
        # Filter layout diffs (OCR bbox positions have natural variance)
        if diff.change_type == "layout":
            # Keep only significant layout changes (page_size, major block shifts)
            # Filter line-level spacing/position diffs
            if subtype in ("layout_shift", "layout_drift", "layout"):
                # Check if this is a major block shift or minor line shift
                # Minor shifts (< 5% of page) are likely OCR variance
                if not md.get("major_block_shift", False):
                    logger.debug("Filtering OCR layout diff: page=%s subtype=%s", 
                               diff.page_num, subtype)
                    continue
        
        # Filter microscopic OCR character changes below threshold.
        # Some producers (e.g. TextComparator) may emit character-change diffs
        # without populating ocr_changed_chars/ratio. In that case, compute
        # significance here so we don't incorrectly treat it as 0-char noise.
        if diff.change_type == "content" and subtype == "character_change":
            changed_chars = md.get("ocr_changed_chars")
            change_ratio = md.get("ocr_change_ratio")

            if (changed_chars is None or change_ratio is None) and ((diff.old_text or "") or (diff.new_text or "")):
                try:
                    from utils.text_normalization import compute_ocr_change_significance

                    sig = compute_ocr_change_significance(diff.old_text or "", diff.new_text or "", ocr=True)
                    changed_chars = sig.get("changed_chars")
                    change_ratio = sig.get("change_ratio")

                    # Cache back onto metadata for downstream tooling.
                    if isinstance(changed_chars, int):
                        md["ocr_changed_chars"] = changed_chars
                    if isinstance(change_ratio, (int, float)):
                        md["ocr_change_ratio"] = float(change_ratio)
                except Exception:
                    # If significance computation fails, fall back to conservative defaults
                    # (do not crash classification).
                    changed_chars = changed_chars if changed_chars is not None else 0
                    change_ratio = change_ratio if change_ratio is not None else 0.0

            changed_chars = int(changed_chars or 0)
            change_ratio = float(change_ratio or 0.0)
            
            # Use settings thresholds
            if changed_chars < settings.ocr_min_change_chars:
                logger.debug("Filtering OCR char diff (chars=%d < %d): page=%s", 
                           changed_chars, settings.ocr_min_change_chars, diff.page_num)
                continue
            
            if change_ratio < settings.ocr_min_change_ratio:
                logger.debug("Filtering OCR char diff (ratio=%.3f < %.3f): page=%s", 
                           change_ratio, settings.ocr_min_change_ratio, diff.page_num)
                continue
        
        # Passed all filters
        filtered.append(diff)
    
    return filtered


def _apply_ocr_adjustments(diff: Diff) -> Diff:
    """
    Apply OCR-specific adjustments to a classified diff.
    
    For OCR diffs:
    - Reduce confidence for formatting/layout changes (unreliable data)
    - Mark punctuation/whitespace changes as low-severity
    - Tag microscopic character changes for potential filtering
    """
    if not _diff_is_ocr(diff):
        return diff
    
    md = diff.metadata or {}
    subtype = md.get("subtype", "")
    
    # For OCR pages, formatting diffs should have very low confidence
    # because OCR font/size data is synthetic
    if diff.change_type == "formatting":
        diff.confidence = min(diff.confidence, 0.3)
        md["ocr_reliability"] = "low"
        md["ocr_note"] = "OCR font/style data is synthetic; diff may be noise"
    
    # Punctuation/whitespace changes from OCR are often noise
    # Mark them for potential filtering in reports
    if subtype in ("punctuation", "whitespace", "punctuation_shift"):
        md["ocr_severity"] = "low"
        md.setdefault("ocr_note", "Common OCR artifact; may be noise")
    
    # Character changes in OCR need to pass significance thresholds
    # The line_comparison module already filters by ocr_min_change_chars/ratio,
    # but we can add additional metadata for downstream
    if subtype == "character_change":
        changed_chars = md.get("ocr_changed_chars", 0)
        change_ratio = md.get("ocr_change_ratio", 0.0)
        
        # Tag as potentially significant or noise based on metrics
        if changed_chars <= 1 or change_ratio < 0.01:
            md["ocr_severity"] = "low"
            md.setdefault("ocr_note", "Single-char difference may be OCR misread")
    
    diff.metadata = md
    return diff


def _diff_is_ocr(diff: Diff) -> bool:
    """Best-effort OCR detection for a diff.

    The pipeline tags diffs with `metadata['is_ocr']`. Keep a fallback so callers
    that build diffs manually can still opt-in.
    """
    try:
        return bool((diff.metadata or {}).get("is_ocr", False))
    except Exception:
        return False


def _classify_single_diff(diff: Diff) -> Diff:
    """Classify a single diff using rule-based heuristics."""
    if diff.metadata is None:
        diff.metadata = {}
    
    # Early subtype assignment for non-content diffs (header/footer, table, figure, layout, visual)
    _ensure_subtype(diff)
    
    # IMPORTANT: text-level changes are CONTENT, not layout
    # Even if bbox moved, text changes are semantic content changes
    subtype = diff.metadata.get("subtype", "")
    if subtype in {"character_change", "punctuation_shift", "whitespace", "semantic_change", "text_modification"}:
        diff.change_type = "content"
        diff = _generate_description(diff)
        return diff
    
    # If change_type is already set and not "content", trust it
    if diff.change_type != "content":
        diff = _generate_description(diff)
        return diff
    
    # Check metadata for formatting hints
    if diff.metadata and "formatting_type" in diff.metadata:
        diff.change_type = "formatting"
        diff = _generate_description(diff)
        return diff
    
    # Rule-based classification
    old_text = diff.old_text or ""
    new_text = diff.new_text or ""
    is_ocr = _diff_is_ocr(diff)

    # Case-only changes are considered formatting, not semantic content.
    # IMPORTANT: This must run before character-difference checks, because
    # `has_character_content_difference()` treats case differences as character changes.
    if _is_case_only_change(old_text, new_text, ocr=is_ocr):
        diff.change_type = "formatting"
        diff.metadata.setdefault("subtype", "case")
        diff = _generate_description(diff)
        return diff
    
    # Check for whitespace-only changes (before punctuation, as whitespace is more common)
    if _is_whitespace_only_change(old_text, new_text, ocr=is_ocr):
        diff.change_type = "formatting"
        diff.metadata.setdefault("subtype", "whitespace")
        diff = _generate_description(diff)
        return diff
    
    # Check for number format changes (e.g., "1,000" vs "1000")
    # Must run before punctuation-only changes, since commas are punctuation.
    if _is_number_format_change(old_text, new_text):
        diff.change_type = "formatting"
        diff.metadata.setdefault("subtype", "number_format")
        diff = _generate_description(diff)
        return diff

    # Check for punctuation-only changes
    if _is_punctuation_only_change(old_text, new_text, ocr=is_ocr):
        diff.change_type = "formatting"
        diff.metadata.setdefault("subtype", "punctuation")
        diff = _generate_description(diff)
        return diff

    # Character changes that are not formatting-only are content changes.
    if _is_character_change_only(old_text, new_text):
        diff.change_type = "content"
        diff.metadata.setdefault("subtype", "character_change")
        diff = _generate_description(diff)
        return diff
    
    # If text content is significantly different, it's a content change
    if diff.diff_type in ("added", "deleted"):
        diff.change_type = "content"
        diff.metadata.setdefault("subtype", "text_addition_deletion")
    elif diff.diff_type == "modified":
        # Check similarity from metadata if available
        similarity = diff.metadata.get("similarity", 0.0)
        if similarity < 0.5:
            diff.change_type = "content"
            diff.metadata.setdefault("subtype", "semantic_change")
        else:
            diff.change_type = "content"
            diff.metadata.setdefault("subtype", "text_modification")
    
    diff = _generate_description(diff)
    return diff


def _is_punctuation_only_change(text_a: str, text_b: str, *, ocr: bool = False) -> bool:
    """Check if the only difference is punctuation."""
    # Remove punctuation and compare using normalization
    text_a_clean = re.sub(r'[^\w\s]', '', text_a)
    text_b_clean = re.sub(r'[^\w\s]', '', text_b)
    return normalize_text(text_a_clean, ocr=ocr) == normalize_text(text_b_clean, ocr=ocr) and text_a != text_b


def _is_whitespace_only_change(text_a: str, text_b: str, *, ocr: bool = False) -> bool:
    """
    Check if the only difference is whitespace.
    
    IMPORTANT: This function assumes character content is the same.
    Always check for character changes BEFORE calling this function.
    """
    # First check if there are character content differences
    # If there are, this is NOT a whitespace-only change
    if has_character_content_difference(text_a, text_b):
        return False
    
    # Use normalization which already handles whitespace normalization
    # If normalized texts are equal but originals differ, it's whitespace-only
    return normalize_text(text_a, ocr=ocr) == normalize_text(text_b, ocr=ocr) and text_a != text_b


def _is_case_only_change(text_a: str, text_b: str, *, ocr: bool = False) -> bool:
    """
    Check if the only difference is letter case.
    
    Note: With text normalization in place, this function is less likely to be triggered
    since case differences are normalized before comparison. Kept for backward compatibility.
    """
    # IMPORTANT:
    # Do not use `normalize_text()` here: it also normalizes whitespace/punctuation,
    # which can incorrectly classify whitespace-only diffs as "case".
    if not text_a and not text_b:
        return False
    if text_a == text_b:
        return False
    return text_a.casefold() == text_b.casefold()


def _is_number_format_change(text_a: str, text_b: str) -> bool:
    """Check if the difference is only in number formatting."""
    # Remove commas, spaces, and compare
    text_a_clean = re.sub(r'[,\s]', '', text_a)
    text_b_clean = re.sub(r'[,\s]', '', text_b)
    # Guard against false positives (e.g. removing a comma in prose).
    if not text_a_clean or not text_b_clean:
        return False
    if not (text_a_clean.isdigit() and text_b_clean.isdigit()):
        return False
    return text_a_clean == text_b_clean and text_a != text_b


def _is_character_change_only(text_a: str, text_b: str) -> bool:
    """
    Check if texts differ in character content (excluding whitespace/punctuation).
    
    This function detects actual character changes (additions, deletions, modifications)
    separate from formatting-only changes (whitespace, punctuation).
    
    Args:
        text_a: First text
        text_b: Second text
    
    Returns:
        True if character content differs, False otherwise
    """
    if not text_a and not text_b:
        return False
    
    # Use the utility function to check for character content differences
    return has_character_content_difference(text_a, text_b)


def _generate_description(diff: Diff) -> Diff:
    """Generate a human-readable description of the diff."""
    description = ""
    metadata = diff.metadata or {}

    scope = metadata.get("scope")
    word_text = metadata.get("word_text")
    scope_prefix = ""
    if scope == "word" and isinstance(word_text, str) and word_text.strip():
        scope_prefix = f"Word '{word_text.strip()}': "
    
    # Handle formatting changes
    formatting_type = metadata.get("formatting_type")
    if formatting_type == "font_size":
        old_size = metadata.get("old_size")
        new_size = metadata.get("new_size")
        if old_size and new_size:
            description = f"{scope_prefix}Font size changed from {old_size:.1f}pt to {new_size:.1f}pt"
    
    elif formatting_type == "font":
        old_font = metadata.get("old_font", "unknown")
        new_font = metadata.get("new_font", "unknown")
        description = f"{scope_prefix}Font changed from '{old_font}' to '{new_font}'"
    
    elif formatting_type == "style":
        old_bold = metadata.get("old_bold", False)
        old_italic = metadata.get("old_italic", False)
        new_bold = metadata.get("new_bold", False)
        new_italic = metadata.get("new_italic", False)
        
        old_style = []
        if old_bold:
            old_style.append("bold")
        if old_italic:
            old_style.append("italic")
        
        new_style = []
        if new_bold:
            new_style.append("bold")
        if new_italic:
            new_style.append("italic")
        
        old_style_str = ", ".join(old_style) if old_style else "normal"
        new_style_str = ", ".join(new_style) if new_style else "normal"
        
        if old_style_str != new_style_str:
            description = f"{scope_prefix}Text style changed from {old_style_str} to {new_style_str}"
    
    elif formatting_type == "color":
        old_color = metadata.get("old_color")
        new_color = metadata.get("new_color")
        if old_color and new_color:
            old_rgb = f"rgb{old_color}"
            new_rgb = f"rgb{new_color}"
            description = f"{scope_prefix}Text color changed from {old_rgb} to {new_rgb}"
    
    elif formatting_type == "spacing":
        old_spacing = metadata.get("old_spacing")
        new_spacing = metadata.get("new_spacing")
        if old_spacing and new_spacing:
            description = f"Spacing changed from {old_spacing:.1f}pt to {new_spacing:.1f}pt"
    
    elif formatting_type == "page_size":
        old_size = metadata.get("old_size")
        new_size = metadata.get("new_size")
        if old_size and new_size:
            old_w, old_h = old_size
            new_w, new_h = new_size
            description = f"Page size changed from {old_w:.0f}x{old_h:.0f}pt to {new_w:.0f}x{new_h:.0f}pt"
    
    # Handle table changes
    table_change = metadata.get("table_change")
    if table_change == "table_added":
        structure = metadata.get("table_structure", {})
        rows = structure.get("rows", "?")
        cols = structure.get("cols", "?")
        description = f"Table added ({rows}x{cols})"
    
    elif table_change == "table_deleted":
        structure = metadata.get("table_structure", {})
        rows = structure.get("rows", "?")
        cols = structure.get("cols", "?")
        description = f"Table removed ({rows}x{cols})"
    
    elif table_change == "structure":
        old_structure = metadata.get("old_structure", {})
        new_structure = metadata.get("new_structure", {})
        old_rows = old_structure.get("rows", "?")
        old_cols = old_structure.get("cols", "?")
        new_rows = new_structure.get("rows", "?")
        new_cols = new_structure.get("cols", "?")
        
        changes = []
        if old_rows != new_rows:
            row_diff = new_rows - old_rows
            changes.append(f"{abs(row_diff)} row{'s' if abs(row_diff) > 1 else ''} {'added' if row_diff > 0 else 'removed'}")
        if old_cols != new_cols:
            col_diff = new_cols - old_cols
            changes.append(f"{abs(col_diff)} column{'s' if abs(col_diff) > 1 else ''} {'added' if col_diff > 0 else 'removed'}")
        
        if changes:
            description = f"Table structure changed: {', '.join(changes)}"
    
    elif table_change == "cell_content":
        row = metadata.get("row", "?")
        col = metadata.get("col", "?")
        description = f"Table cell changed (row {row}, col {col})"
    
    # Handle header/footer changes
    header_footer_change = metadata.get("header_footer_change")
    if header_footer_change:
        if diff.diff_type == "added":
            description = f"{header_footer_change.capitalize()} added"
        elif diff.diff_type == "deleted":
            description = f"{header_footer_change.capitalize()} removed"
        elif diff.diff_type == "modified":
            description = f"{header_footer_change.capitalize()} changed"
    
    # Handle figure changes
    figure_change = metadata.get("figure_change")
    if figure_change == "numbering":
        old_number = metadata.get("old_number")
        new_number = metadata.get("new_number")
        if old_number and new_number:
            description = f"Figure numbering changed from {old_number} to {new_number}"
    
    elif figure_change == "caption_text":
        description = "Figure caption text changed"
    
    elif figure_change == "figure_added":
        description = "Figure added"
    
    elif figure_change == "figure_deleted":
        description = "Figure removed"
    
    # Handle text changes
    if not description:
        subtype = metadata.get("subtype")
        if subtype == "character_change":
            description = "Character content changed"
        elif subtype in ("punctuation", "punctuation_shift"):
            description = "Punctuation changed"
        elif subtype == "whitespace":
            description = "Whitespace changed"
        elif subtype == "case":
            description = "Text case changed"
        elif subtype == "number_format":
            description = "Number formatting changed"
        elif diff.diff_type == "added":
            preview = (diff.new_text or "")[:50]
            if preview:
                description = f"Text added: '{preview}{'...' if len(diff.new_text or '') > 50 else ''}'"
            else:
                description = "Content added"
        elif diff.diff_type == "deleted":
            preview = (diff.old_text or "")[:50]
            if preview:
                description = f"Text removed: '{preview}{'...' if len(diff.old_text or '') > 50 else ''}'"
            else:
                description = "Content removed"
        elif diff.diff_type == "modified":
            old_preview = (diff.old_text or "")[:30]
            new_preview = (diff.new_text or "")[:30]
            if old_preview and new_preview:
                description = f"Text changed: '{old_preview}{'...' if len(diff.old_text or '') > 30 else ''}' → '{new_preview}{'...' if len(diff.new_text or '') > 30 else ''}'"
            else:
                description = "Text modified"
    
    # Store description in metadata
    if description:
        diff.metadata["description"] = description
    
    return diff


def get_diff_summary(diffs: List[Diff]) -> dict:
    """Generate a summary of diff classifications."""
    summary = {
        "total": len(diffs),
        "by_type": {},
        "by_change_type": {},
        "by_subtype": {},
        "ocr_stats": {
            "ocr_diffs": 0,
            "ocr_low_severity": 0,
            "ocr_formatting_noise": 0,
        },
    }
    
    for diff in diffs:
        # Count by diff_type
        summary["by_type"][diff.diff_type] = summary["by_type"].get(diff.diff_type, 0) + 1
        
        # Count by change_type
        summary["by_change_type"][diff.change_type] = summary["by_change_type"].get(diff.change_type, 0) + 1
        
        # Count by subtype
        subtype = diff.metadata.get("subtype", "unknown")
        summary["by_subtype"][subtype] = summary["by_subtype"].get(subtype, 0) + 1
        
        # OCR-specific stats
        if _diff_is_ocr(diff):
            summary["ocr_stats"]["ocr_diffs"] += 1
            md = diff.metadata or {}
            if md.get("ocr_severity") == "low":
                summary["ocr_stats"]["ocr_low_severity"] += 1
            if md.get("ocr_reliability") == "low" and diff.change_type == "formatting":
                summary["ocr_stats"]["ocr_formatting_noise"] += 1
    
    return summary
