"""Header and footer detection and comparison."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from comparison.models import Diff, PageData
from utils.logging import logger
from utils.text_normalization import normalize_text


_DIGIT_RE = re.compile(r"\d+")


def _canon_hf_text(text: str | None, *, ocr: bool = False) -> str:
    """Group key: normalize + digits -> {#}."""
    if not text:
        return ""
    t = normalize_text(text, ocr=ocr)
    t = " ".join(t.split())
    return _DIGIT_RE.sub("{#}", t)


def _tpl_hf_text(text: str | None) -> str | None:
    """Display template: keep original, digits -> {#}."""
    if text is None:
        return None
    return _DIGIT_RE.sub("{#}", text.strip())


def _aggregate_header_footer_diffs(diffs: List[Diff], *, min_pages: int = 2) -> List[Diff]:
    """
    Aggregate repeating header/footer diffs across pages into single document-level diffs.
    
    Args:
        diffs: List of header/footer Diff objects
        min_pages: Minimum number of pages for a pattern to be considered repeating
    
    Returns:
        Aggregated list of Diff objects
    """
    groups: Dict[tuple, List[Diff]] = defaultdict(list)
    keep: List[Diff] = []

    for d in diffs:
        md = d.metadata or {}
        kind = md.get("header_footer_change")  # "header" | "footer"
        if not kind:
            keep.append(d)
            continue
        # Use OCR flag from metadata if available
        is_ocr = md.get("is_ocr_page", False)
        key = (kind, d.diff_type, _canon_hf_text(d.old_text, ocr=is_ocr), _canon_hf_text(d.new_text, ocr=is_ocr))
        groups[key].append(d)

    out: List[Diff] = []
    for (kind, diff_type, old_key, new_key), items in groups.items():
        if len(items) < min_pages:
            out.extend(items)
            continue

        items.sort(key=lambda x: x.page_num)
        rep = items[0]
        pages = [x.page_num for x in items]

        md = dict(rep.metadata or {})
        md.update({
            "aggregated": True,
            "scope": "document",
            "pages": pages,
            "count": len(items),
            "text_template": {"old": _tpl_hf_text(rep.old_text), "new": _tpl_hf_text(rep.new_text)},
            # Full occurrences for rendering across all pages:
            "occurrences": [
                {"page_num": x.page_num, "bbox": x.bbox, "old_text": x.old_text, "new_text": x.new_text}
                for x in items
            ],
        })

        avg_conf = sum(x.confidence for x in items) / len(items)

        out.append(Diff(
            page_num=rep.page_num,         # "anchor" to first page
            diff_type=rep.diff_type,
            change_type=rep.change_type,   # stays "formatting"
            old_text=_tpl_hf_text(rep.old_text) or rep.old_text,
            new_text=_tpl_hf_text(rep.new_text) or rep.new_text,
            bbox=rep.bbox,                 # first page bbox (full list in metadata.occurrences)
            confidence=min(1.0, avg_conf + 0.05),
            metadata=md,
        ))

    return keep + out


def _is_ocr_page(page: PageData) -> bool:
    """Best-effort detection of OCR-derived pages based on extraction metadata."""
    extraction_method = str(page.metadata.get("extraction_method", "")).lower()
    line_extraction_method = str(page.metadata.get("line_extraction_method", "")).lower()
    ocr_engine_used = str(page.metadata.get("ocr_engine_used", "")).lower()
    return (
        "ocr" in extraction_method
        or "ocr" in line_extraction_method
        or "ocr" in ocr_engine_used
        or "tesseract" in ocr_engine_used
        or "paddle" in ocr_engine_used
        or "deepseek" in ocr_engine_used
    )


@dataclass
class HeaderFooter:
    """Represents a header or footer."""
    text: str
    bbox: Dict[str, float]  # Bounding box
    page_num: int
    is_header: bool
    is_page_number: bool = False
    is_repeating: bool = field(default=False)


def detect_headers_footers(pages: List[PageData]) -> Dict[int, Tuple[List[HeaderFooter], List[HeaderFooter]]]:
    """
    Detect headers and footers across all pages.
    
    Args:
        pages: List of PageData objects
    
    Returns:
        Dictionary mapping page_num to (headers, footers) tuple
    """
    logger.info("Detecting headers and footers across %d pages", len(pages))
    
    result: Dict[int, Tuple[List[HeaderFooter], List[HeaderFooter]]] = {}
    page_is_ocr: Dict[int, bool] = {}
    
    for idx, page in enumerate(pages):
        page_is_ocr[page.page_num] = _is_ocr_page(page)
        # Debug log for first page
        if idx == 0 and len(page.blocks) > 0:
            top_blocks = sorted(page.blocks, key=lambda b: b.bbox["y"])[:3]
            logger.debug(
                "Page %d: %d blocks available. Top 3 block snippets: %s",
                page.page_num,
                len(page.blocks),
                [b.text[:50] + "..." if len(b.text) > 50 else b.text for b in top_blocks]
            )
        headers = _detect_headers(page)
        footers = _detect_footers(page)
        result[page.page_num] = (headers, footers)
    
    # Identify repeating headers/footers across pages
    _identify_repeating_patterns(result, page_is_ocr)
    
    logger.debug("Header/footer detection complete")
    return result


def _detect_headers(page: PageData) -> List[HeaderFooter]:
    """Detect headers on a single page."""
    from config.settings import settings
    
    headers = []
    
    # Early warning: no blocks available
    if len(page.blocks) == 0:
        logger.warning(
            "No blocks on page %d; header/footer detection skipped. "
            "OCR engine: %s, extraction_method: %s",
            page.page_num,
            page.metadata.get("ocr_engine_used", "unknown"),
            page.metadata.get("extraction_method", "unknown")
        )
        return headers
    
    # Header region is typically top 10% of page (configurable)
    header_region_height = page.height * settings.header_region_height_ratio
    header_threshold_y = header_region_height
    
    for block in page.blocks:
        block_y = block.bbox["y"]
        
        # Check if block is in header region (top of page)
        if block_y < header_threshold_y and block.text.strip():
            # Check if it looks like a page number pattern
            is_page_num = _is_page_number_pattern(block.text)
            
            headers.append(HeaderFooter(
                text=block.text.strip(),
                bbox=block.bbox.copy(),
                page_num=page.page_num,
                is_header=True,
                is_page_number=is_page_num,
            ))
    
    return headers


def _detect_footers(page: PageData) -> List[HeaderFooter]:
    """Detect footers on a single page."""
    from config.settings import settings
    
    footers = []
    
    # Early warning: no blocks available (only log once per page, already logged in headers)
    if len(page.blocks) == 0:
        return footers
    
    # Footer region is typically bottom 10% of page (configurable)
    footer_region_height = page.height * settings.footer_region_height_ratio
    footer_threshold_y = page.height - footer_region_height
    
    for block in page.blocks:
        block_y = block.bbox["y"]
        block_height = block.bbox["height"]
        
        # Check if block is in footer region (bottom of page)
        if block_y + block_height > footer_threshold_y and block.text.strip():
            # Check if it looks like a page number pattern
            is_page_num = _is_page_number_pattern(block.text)
            
            footers.append(HeaderFooter(
                text=block.text.strip(),
                bbox=block.bbox.copy(),
                page_num=page.page_num,
                is_header=False,
                is_page_number=is_page_num,
            ))
    
    return footers


def _is_page_number_pattern(text: str) -> bool:
    """Check if text matches common page number patterns."""
    text_lower = text.lower().strip()
    
    # Common patterns: "Page 1", "1 / 10", "1 of 10", "1-10", just numbers
    patterns = [
        r'^page\s+\d+',  # "Page 1", "Page 10"
        r'^\d+\s*[/-]\s*\d+',  # "1/10", "1 - 10"
        r'^\d+\s+of\s+\d+',  # "1 of 10"
        r'^\d+$',  # Just a number
    ]
    
    for pattern in patterns:
        if re.match(pattern, text_lower):
            return True
    
    return False


def _identify_repeating_patterns(
    header_footer_map: Dict[int, Tuple[List[HeaderFooter], List[HeaderFooter]]],
    page_is_ocr: Dict[int, bool],
) -> None:
    """
    Identify headers/footers that repeat across pages.
    
    This helps distinguish document headers/footers from page-specific content.
    """
    if len(header_footer_map) < 2:
        return
    
    # Count occurrences of each header/footer text
    header_text_counts: Dict[str, int] = {}
    footer_text_counts: Dict[str, int] = {}
    
    for page_num, (headers, footers) in header_footer_map.items():
        is_ocr_page = page_is_ocr.get(page_num, False)
        for header in headers:
            if not header.is_page_number:  # Skip page numbers
                text = normalize_text(header.text, ocr=is_ocr_page)
                header_text_counts[text] = header_text_counts.get(text, 0) + 1
        
        for footer in footers:
            if not footer.is_page_number:  # Skip page numbers
                text = normalize_text(footer.text, ocr=is_ocr_page)
                footer_text_counts[text] = footer_text_counts.get(text, 0) + 1
    
    # Mark repeating headers/footers (appear on multiple pages)
    from config.settings import settings
    min_repetition_ratio = settings.header_footer_repetition_threshold
    threshold = max(2, int(len(header_footer_map) * min_repetition_ratio))
    
    for page_num, (headers, footers) in header_footer_map.items():
        is_ocr_page = page_is_ocr.get(page_num, False)
        for header in headers:
            if not header.is_page_number:
                text = normalize_text(header.text, ocr=is_ocr_page)
                if header_text_counts.get(text, 0) >= threshold:
                    header.is_repeating = True
        
        for footer in footers:
            if not footer.is_page_number:
                text = normalize_text(footer.text, ocr=is_ocr_page)
                if footer_text_counts.get(text, 0) >= threshold:
                    footer.is_repeating = True


def compare_headers_footers(
    pages_a: List[PageData],
    pages_b: List[PageData],
    alignment_map: dict | None = None,
) -> List[Diff]:
    """
    Compare headers and footers between two documents.
    
    Args:
        pages_a: Pages from first document
        pages_b: Pages from second document
        alignment_map: Optional page alignment map
    
    Returns:
        List of Diff objects representing header/footer changes
    """
    logger.info("Comparing headers and footers")
    
    from comparison.alignment import align_pages
    
    if alignment_map is None:
        alignment_map = align_pages(pages_a, pages_b, use_similarity=False)
    
    # Detect headers/footers in both documents
    headers_footers_a = detect_headers_footers(pages_a)
    headers_footers_b = detect_headers_footers(pages_b)
    
    all_diffs: List[Diff] = []
    page_b_lookup = {page.page_num: page for page in pages_b}
    
    for page_a in pages_a:
        if page_a.page_num not in alignment_map:
            continue
        
        page_b_num, confidence = alignment_map[page_a.page_num]
        if page_b_num not in page_b_lookup:
            continue
        
        page_b = page_b_lookup[page_b_num]

        is_ocr_page = _is_ocr_page(page_a) or _is_ocr_page(page_b)
        
        # Get headers/footers for both pages
        headers_a, footers_a = headers_footers_a.get(page_a.page_num, ([], []))
        headers_b, footers_b = headers_footers_b.get(page_b_num, ([], []))
        
        # Compare headers
        header_diffs = _compare_header_footer_lists(
            headers_a, headers_b, page_a.page_num, "header",
            page_a.width, page_a.height, confidence,
            is_ocr_page=is_ocr_page,
        )
        all_diffs.extend(header_diffs)
        
        # Compare footers
        footer_diffs = _compare_header_footer_lists(
            footers_a, footers_b, page_a.page_num, "footer",
            page_a.width, page_a.height, confidence,
            is_ocr_page=is_ocr_page,
        )
        all_diffs.extend(footer_diffs)
    
    raw_count = len(all_diffs)
    logger.info("Header/Footer Comparison: %d raw diffs", raw_count)

    # Aggregate repeating header/footer patterns into document-level diffs
    all_diffs = _aggregate_header_footer_diffs(all_diffs, min_pages=2)
    
    logger.info("Detected %d header/footer differences (after aggregation, was %d)", len(all_diffs), raw_count)
    return all_diffs


def _compare_header_footer_lists(
    items_a: List[HeaderFooter],
    items_b: List[HeaderFooter],
    page_num: int,
    item_type: str,
    page_width: float,
    page_height: float,
    confidence: float,
    *,
    is_ocr_page: bool = False,
) -> List[Diff]:
    """Compare two lists of headers or footers."""
    from comparison.models import Diff
    from utils.coordinates import normalize_bbox
    
    diffs = []
    
    # Match items by position and text similarity
    matched_pairs, unmatched_a, unmatched_b = _match_header_footer_items(
        items_a, items_b, is_ocr_page=is_ocr_page
    )
    
    # Compare matched items using normalized comparison
    for item_a, item_b in matched_pairs:
        if normalize_text(item_a.text, ocr=is_ocr_page) != normalize_text(item_b.text, ocr=is_ocr_page):
            normalized_bbox = normalize_bbox(
                (item_a.bbox["x"], item_a.bbox["y"],
                 item_a.bbox["x"] + item_a.bbox["width"],
                 item_a.bbox["y"] + item_a.bbox["height"]),
                page_width, page_height
            )
            
            diffs.append(Diff(
                page_num=page_num,
                diff_type="modified",
                change_type="formatting",
                old_text=item_a.text,
                new_text=item_b.text,
                bbox=normalized_bbox,
                confidence=confidence,
                metadata={
                    "header_footer_change": item_type,
                    "is_page_number": item_a.is_page_number,
                    "is_repeating": item_a.is_repeating or item_b.is_repeating,
                    "page_width": page_width,
                    "page_height": page_height,
                },
            ))
    
    # Unmatched items in doc_a are deleted
    for item in unmatched_a:
        normalized_bbox = normalize_bbox(
            (item.bbox["x"], item.bbox["y"],
             item.bbox["x"] + item.bbox["width"],
             item.bbox["y"] + item.bbox["height"]),
            page_width, page_height
        )
        
        diffs.append(Diff(
            page_num=page_num,
            diff_type="deleted",
            change_type="formatting",
            old_text=item.text,
            new_text=None,
            bbox=normalized_bbox,
            confidence=confidence,
            metadata={
                "header_footer_change": item_type,
                "is_page_number": item.is_page_number,
                "is_repeating": item.is_repeating,
                "page_width": page_width,
                "page_height": page_height,
            },
        ))
    
    # Unmatched items in doc_b are added
    for item in unmatched_b:
        normalized_bbox = normalize_bbox(
            (item.bbox["x"], item.bbox["y"],
             item.bbox["x"] + item.bbox["width"],
             item.bbox["y"] + item.bbox["height"]),
            page_width, page_height
        )
        
        diffs.append(Diff(
            page_num=page_num,
            diff_type="added",
            change_type="formatting",
            old_text=None,
            new_text=item.text,
            bbox=normalized_bbox,
            confidence=confidence,
            metadata={
                "header_footer_change": item_type,
                "is_page_number": item.is_page_number,
                "is_repeating": item.is_repeating,
                "page_width": page_width,
                "page_height": page_height,
            },
        ))
    
    return diffs


def _match_header_footer_items(
    items_a: List[HeaderFooter],
    items_b: List[HeaderFooter],
    *,
    is_ocr_page: bool = False,
) -> Tuple[List[Tuple[HeaderFooter, HeaderFooter]], List[HeaderFooter], List[HeaderFooter]]:
    """Match headers/footers between two pages."""
    matched: List[Tuple[HeaderFooter, HeaderFooter]] = []
    unmatched_a = items_a.copy()
    unmatched_b = items_b.copy()
    
    # Match by position similarity and text similarity
    for item_a in items_a:
        best_match = None
        best_score = 0.0
        
        for item_b in unmatched_b:
            # Calculate similarity score
            pos_score = _calculate_position_similarity(item_a.bbox, item_b.bbox)
            # Use normalized comparison for text similarity
            text_score = (
                1.0
                if normalize_text(item_a.text, ocr=is_ocr_page)
                == normalize_text(item_b.text, ocr=is_ocr_page)
                else 0.0
            )
            combined_score = (pos_score * 0.5 + text_score * 0.5)
            
            from config.settings import settings
            if combined_score > best_score and combined_score > settings.header_footer_match_threshold:
                best_score = combined_score
                best_match = item_b
        
        if best_match:
            matched.append((item_a, best_match))
            unmatched_a.remove(item_a)
            unmatched_b.remove(best_match)
    
    return matched, unmatched_a, unmatched_b


def _calculate_position_similarity(bbox_a: Dict[str, float], bbox_b: Dict[str, float]) -> float:
    """Calculate position similarity between two bounding boxes."""
    # Compare x-coordinate (headers/footers should align horizontally)
    x_a = bbox_a["x"] + bbox_a["width"] / 2
    x_b = bbox_b["x"] + bbox_b["width"] / 2
    
    # Compare y-coordinate (headers should be at top, footers at bottom)
    y_a = bbox_a["y"] + bbox_a["height"] / 2
    y_b = bbox_b["y"] + bbox_b["height"] / 2
    
    # Normalize differences (assume page width/height ~600pt)
    x_diff = abs(x_a - x_b) / 600.0
    y_diff = abs(y_a - y_b) / 600.0
    
    # Similarity decreases with distance
    similarity = 1.0 - min(1.0, (x_diff + y_diff) / 2.0)
    
    return max(0.0, similarity)

