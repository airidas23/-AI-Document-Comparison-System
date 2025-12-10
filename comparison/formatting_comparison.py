"""Style/layout difference detection."""
from __future__ import annotations

from typing import List

from comparison.alignment import align_pages, align_sections
from comparison.models import Diff, PageData
from config.settings import settings
from utils.logging import logger
from utils.text_normalization import normalize_text


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
        
        # Compare formatting for aligned blocks
        for idx_a, idx_b in block_alignment.items():
            if idx_a >= len(page_a.blocks) or idx_b >= len(page_b.blocks):
                continue
            
            block_a = page_a.blocks[idx_a]
            block_b = page_b.blocks[idx_b]
            
            # Skip if text content is different (handled by text comparison)
            # Use normalized comparison to ignore case and minor differences
            if normalize_text(block_a.text) != normalize_text(block_b.text):
                continue
            
            # Compare styles (pass page dimensions for normalization)
            style_diffs = _compare_styles(
                block_a, block_b, page_a.page_num, confidence,
                page_a.width, page_a.height
            )
            all_diffs.extend(style_diffs)
        
        # Compare page-level layout
        layout_diffs = _compare_page_layout(page_a, page_b, confidence)
        all_diffs.extend(layout_diffs)
    
    logger.info("Detected %d formatting differences", len(all_diffs))
    return all_diffs


def _compare_styles(
    block_a, block_b, page_num: int, confidence: float,
    page_width: float, page_height: float
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
    
    # Check extraction method - handle OCR-extracted styles
    extraction_method_a = getattr(block_a, 'metadata', {}).get("extraction_method", "")
    extraction_method_b = getattr(block_b, 'metadata', {}).get("extraction_method", "")
    
    # Skip formatting comparison for OCR if configured
    is_ocr_a = "ocr" in extraction_method_a.lower()
    is_ocr_b = "ocr" in extraction_method_b.lower()
    
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
