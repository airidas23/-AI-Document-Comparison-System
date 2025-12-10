"""Classify diffs into content vs formatting types."""
from __future__ import annotations

import re
from typing import List

from comparison.models import ChangeType, Diff, DiffType
from utils.logging import logger
from utils.text_normalization import normalize_text
from utils.text_diff import has_character_content_difference


def classify_diffs(diffs: List[Diff]) -> List[Diff]:
    """
    Classify differences using rule-based heuristics.
    
    Categorizes diffs into subtypes:
    - content: semantic changes, paraphrasing, additions/deletions
    - formatting: font, size, color, style changes
    - layout: spacing, page size, column changes
    - visual: pixel-level differences
    
    Args:
        diffs: List of unclassified Diff objects
    
    Returns:
        List of Diff objects with updated change_type and metadata
    """
    logger.info("Classifying %d diffs", len(diffs))
    
    classified = []
    for diff in diffs:
        classified_diff = _classify_single_diff(diff)
        classified.append(classified_diff)
    
    logger.debug("Classification complete: %d diffs processed", len(classified))
    return classified


def _classify_single_diff(diff: Diff) -> Diff:
    """Classify a single diff using rule-based heuristics."""
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
    
    # PRIORITY: Check for character changes FIRST (before whitespace/punctuation checks)
    # This ensures character changes are always classified as content, not formatting
    if _is_character_change_only(old_text, new_text):
        diff.change_type = "content"
        # Check if metadata already has subtype from text comparison
        if "subtype" not in diff.metadata or diff.metadata.get("subtype") != "character_change":
            diff.metadata["subtype"] = "character_change"
        diff = _generate_description(diff)
        return diff
    
    # Check for whitespace-only changes (before punctuation, as whitespace is more common)
    if _is_whitespace_only_change(old_text, new_text):
        diff.change_type = "formatting"
        diff.metadata["subtype"] = "whitespace"
        diff = _generate_description(diff)
        return diff
    
    # Check for punctuation-only changes
    if _is_punctuation_only_change(old_text, new_text):
        diff.change_type = "formatting"
        diff.metadata["subtype"] = "punctuation"
        diff = _generate_description(diff)
        return diff
    
    # Check for case-only changes
    # Note: This is less likely to trigger now since text is normalized before comparison,
    # but kept for backward compatibility and edge cases
    if _is_case_only_change(old_text, new_text):
        diff.change_type = "formatting"
        diff.metadata["subtype"] = "case"
        diff = _generate_description(diff)
        return diff
    
    # Check for number format changes (e.g., "1,000" vs "1000")
    if _is_number_format_change(old_text, new_text):
        diff.change_type = "formatting"
        diff.metadata["subtype"] = "number_format"
        diff = _generate_description(diff)
        return diff
    
    # If text content is significantly different, it's a content change
    if diff.diff_type in ("added", "deleted"):
        diff.change_type = "content"
        diff.metadata["subtype"] = "text_addition_deletion"
    elif diff.diff_type == "modified":
        # Check similarity from metadata if available
        similarity = diff.metadata.get("similarity", 0.0)
        if similarity < 0.5:
            diff.change_type = "content"
            diff.metadata["subtype"] = "semantic_change"
        else:
            diff.change_type = "content"
            diff.metadata["subtype"] = "text_modification"
    
    diff = _generate_description(diff)
    return diff


def _is_punctuation_only_change(text_a: str, text_b: str) -> bool:
    """Check if the only difference is punctuation."""
    # Remove punctuation and compare using normalization
    text_a_clean = re.sub(r'[^\w\s]', '', text_a)
    text_b_clean = re.sub(r'[^\w\s]', '', text_b)
    return normalize_text(text_a_clean) == normalize_text(text_b_clean) and text_a != text_b


def _is_whitespace_only_change(text_a: str, text_b: str) -> bool:
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
    return normalize_text(text_a) == normalize_text(text_b) and text_a != text_b


def _is_case_only_change(text_a: str, text_b: str) -> bool:
    """
    Check if the only difference is letter case.
    
    Note: With text normalization in place, this function is less likely to be triggered
    since case differences are normalized before comparison. Kept for backward compatibility.
    """
    return normalize_text(text_a) == normalize_text(text_b) and text_a != text_b


def _is_number_format_change(text_a: str, text_b: str) -> bool:
    """Check if the difference is only in number formatting."""
    # Remove commas, spaces, and compare
    text_a_clean = re.sub(r'[,\s]', '', text_a)
    text_b_clean = re.sub(r'[,\s]', '', text_b)
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
    
    # Handle formatting changes
    formatting_type = metadata.get("formatting_type")
    if formatting_type == "font_size":
        old_size = metadata.get("old_size")
        new_size = metadata.get("new_size")
        if old_size and new_size:
            description = f"Font size changed from {old_size:.1f}pt to {new_size:.1f}pt"
    
    elif formatting_type == "font":
        old_font = metadata.get("old_font", "unknown")
        new_font = metadata.get("new_font", "unknown")
        description = f"Font changed from '{old_font}' to '{new_font}'"
    
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
            description = f"Text style changed from {old_style_str} to {new_style_str}"
    
    elif formatting_type == "color":
        old_color = metadata.get("old_color")
        new_color = metadata.get("new_color")
        if old_color and new_color:
            old_rgb = f"rgb{old_color}"
            new_rgb = f"rgb{new_color}"
            description = f"Text color changed from {old_rgb} to {new_rgb}"
    
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
        elif subtype == "punctuation":
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
    }
    
    for diff in diffs:
        # Count by diff_type
        summary["by_type"][diff.diff_type] = summary["by_type"].get(diff.diff_type, 0) + 1
        
        # Count by change_type
        summary["by_change_type"][diff.change_type] = summary["by_change_type"].get(diff.change_type, 0) + 1
        
        # Count by subtype
        subtype = diff.metadata.get("subtype", "unknown")
        summary["by_subtype"][subtype] = summary["by_subtype"].get(subtype, 0) + 1
    
    return summary
