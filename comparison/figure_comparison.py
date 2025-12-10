"""Figure caption and numbering comparison."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from comparison.alignment import align_pages
from comparison.models import Diff, PageData
from utils.coordinates import normalize_bbox
from utils.logging import logger
from utils.text_normalization import normalize_text


@dataclass
class FigureCaption:
    """Represents a figure with its caption."""
    figure_bbox: Dict[str, float]
    caption_text: str
    page_num: int
    caption_number: Optional[int] = None
    caption_label: Optional[str] = None  # "Figure", "Fig.", etc.
    caption_bbox: Optional[Dict[str, float]] = None


def extract_figure_captions(page: PageData) -> List[FigureCaption]:
    """
    Extract figure captions from a page.
    
    Captions are typically found:
    - Below figures (most common)
    - Above figures (less common)
    - As part of figure bounding box
    
    Args:
        page: PageData object with figures in metadata
    
    Returns:
        List of FigureCaption objects
    """
    figures = page.metadata.get("figures", [])
    if not figures:
        return []
    
    captions = []
    
    for figure in figures:
        fig_bbox = figure.get("bbox", [])
        if not fig_bbox or len(fig_bbox) < 4:
            continue
        
        # Convert bbox to dict format if needed
        if isinstance(fig_bbox, list):
            fig_bbox_dict = {
                "x": fig_bbox[0],
                "y": fig_bbox[1],
                "width": fig_bbox[2] - fig_bbox[0] if len(fig_bbox) > 2 else 0,
                "height": fig_bbox[3] - fig_bbox[1] if len(fig_bbox) > 3 else 0,
            }
        else:
            fig_bbox_dict = fig_bbox
        
        # Find caption near the figure (typically below)
        caption = _find_caption_near_figure(page, fig_bbox_dict)
        
        if caption:
            captions.append(FigureCaption(
                figure_bbox=fig_bbox_dict,
                caption_text=caption["text"],
                page_num=page.page_num,
                caption_number=caption.get("number"),
                caption_label=caption.get("label"),
                caption_bbox=caption.get("bbox"),
            ))
        else:
            # Figure without caption
            captions.append(FigureCaption(
                figure_bbox=fig_bbox_dict,
                caption_text="",
                page_num=page.page_num,
            ))
    
    return captions


def _find_caption_near_figure(page: PageData, figure_bbox: Dict[str, float]) -> Optional[Dict]:
    """
    Find caption text near a figure.
    
    Looks for text blocks below (or above) the figure that match caption patterns.
    """
    fig_x = figure_bbox["x"]
    fig_y = figure_bbox["y"]
    fig_width = figure_bbox["width"]
    fig_height = figure_bbox["height"]
    fig_bottom = fig_y + fig_height
    
    # Search region: below figure, within reasonable distance
    from config.settings import settings
    search_margin = settings.caption_search_margin
    search_bottom = fig_bottom + settings.caption_search_distance
    
    # Find text blocks in the search region
    candidates = []
    
    for block in page.blocks:
        block_y = block.bbox["y"]
        block_bottom = block_y + block.bbox["height"]
        block_x = block.bbox["x"]
        block_width = block.bbox["width"]
        
        # Check if block is in search region (horizontally aligned with figure)
        horizontal_overlap = not (block_x + block_width < fig_x or block_x > fig_x + fig_width)
        
        # Check if block is below figure (within search margin)
        is_below = fig_bottom <= block_y <= search_bottom
        
        if horizontal_overlap and is_below:
            text = block.text.strip()
            
            # Check if text matches caption pattern
            caption_match = _match_caption_pattern(text)
            if caption_match:
                candidates.append({
                    "text": text,
                    "bbox": block.bbox.copy(),
                    "y": block_y,
                    "number": caption_match.get("number"),
                    "label": caption_match.get("label"),
                })
    
    # Return the closest candidate (lowest y coordinate = closest to figure)
    if candidates:
        candidates.sort(key=lambda c: c["y"])
        return candidates[0]
    
    return None


def _match_caption_pattern(text: str) -> Optional[Dict]:
    """
    Match text against common figure caption patterns.
    
    Patterns:
    - "Figure 1: Description"
    - "Fig. 1. Description"
    - "Figure 1.1: Description"
    - "Figure 3-2: Description"
    """
    text_lower = text.lower().strip()
    
    # Pattern: "Figure N" or "Fig. N" followed by optional description
    patterns = [
        (r'^(figure|fig\.?)\s+(\d+(?:[-.]\d+)?)\s*[:.]?\s*(.*)$', "Figure"),
        (r'^(fig\.?)\s+(\d+(?:[-.]\d+)?)\s*[:.]?\s*(.*)$', "Fig."),
    ]
    
    for pattern, label in patterns:
        match = re.match(pattern, text_lower, re.IGNORECASE)
        if match:
            number_str = match.group(2).replace("-", ".").replace(".", ".")
            # Extract first number for comparison
            number_match = re.search(r'^(\d+)', number_str)
            if number_match:
                number = int(number_match.group(1))
                return {
                    "number": number,
                    "label": label,
                    "full_number": number_str,
                }
    
    return None


def compare_figure_captions(
    pages_a: List[PageData],
    pages_b: List[PageData],
    alignment_map: dict | None = None,
) -> List[Diff]:
    """
    Compare figure captions and numbering between two documents.
    
    Detects:
    - Caption text changes
    - Figure numbering changes (e.g., "Figure 5" → "Figure 6")
    - Missing/added captions
    
    Args:
        pages_a: Pages from first document
        pages_b: Pages from second document
        alignment_map: Optional page alignment map
    
    Returns:
        List of Diff objects representing figure caption changes
    """
    logger.info("Comparing figure captions")
    
    from comparison.alignment import align_pages
    
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
        
        # Extract figure captions from both pages
        captions_a = extract_figure_captions(page_a)
        captions_b = extract_figure_captions(page_b)
        
        # Match figures between pages
        matched_pairs, unmatched_a, unmatched_b = _match_figures(captions_a, captions_b)
        
        # Compare matched figures
        for caption_a, caption_b in matched_pairs:
            caption_diffs = _compare_caption_pair(
                caption_a, caption_b, page_a.page_num, confidence,
                page_a.width, page_a.height
            )
            all_diffs.extend(caption_diffs)
        
        # Unmatched figures in doc_a
        for caption in unmatched_a:
            normalized_bbox = normalize_bbox(
                (caption.figure_bbox["x"], caption.figure_bbox["y"],
                 caption.figure_bbox["x"] + caption.figure_bbox["width"],
                 caption.figure_bbox["y"] + caption.figure_bbox["height"]),
                page_a.width, page_a.height
            )
            
            all_diffs.append(Diff(
                page_num=page_a.page_num,
                diff_type="deleted",
                change_type="content",
                old_text=caption.caption_text or "Figure",
                new_text=None,
                bbox=normalized_bbox,
                confidence=confidence,
                metadata={
                    "figure_change": "figure_deleted",
                    "caption_number": caption.caption_number,
                    "page_width": page_a.width,
                    "page_height": page_a.height,
                },
            ))
        
        # Unmatched figures in doc_b
        for caption in unmatched_b:
            normalized_bbox = normalize_bbox(
                (caption.figure_bbox["x"], caption.figure_bbox["y"],
                 caption.figure_bbox["x"] + caption.figure_bbox["width"],
                 caption.figure_bbox["y"] + caption.figure_bbox["height"]),
                page_b.width, page_b.height
            )
            
            all_diffs.append(Diff(
                page_num=page_b.page_num,
                diff_type="added",
                change_type="content",
                old_text=None,
                new_text=caption.caption_text or "Figure",
                bbox=normalized_bbox,
                confidence=confidence,
                metadata={
                    "figure_change": "figure_added",
                    "caption_number": caption.caption_number,
                    "page_width": page_b.width,
                    "page_height": page_b.height,
                },
            ))
    
    logger.info("Detected %d figure caption differences", len(all_diffs))
    return all_diffs


def _match_figures(
    captions_a: List[FigureCaption],
    captions_b: List[FigureCaption],
) -> Tuple[List[Tuple[FigureCaption, FigureCaption]], List[FigureCaption], List[FigureCaption]]:
    """Match figures between two pages based on position."""
    matched: List[Tuple[FigureCaption, FigureCaption]] = []
    unmatched_a = captions_a.copy()
    unmatched_b = captions_b.copy()
    
    # Match by position overlap
    for caption_a in captions_a:
        best_match = None
        best_score = 0.0
        
        for caption_b in unmatched_b:
            from config.settings import settings
            score = _calculate_figure_overlap(caption_a, caption_b)
            if score > best_score and score > settings.figure_overlap_threshold:
                best_score = score
                best_match = caption_b
        
        if best_match:
            matched.append((caption_a, best_match))
            unmatched_a.remove(caption_a)
            unmatched_b.remove(best_match)
    
    return matched, unmatched_a, unmatched_b


def _calculate_figure_overlap(caption_a: FigureCaption, caption_b: FigureCaption) -> float:
    """Calculate overlap score between two figures."""
    bbox_a = caption_a.figure_bbox
    bbox_b = caption_b.figure_bbox
    
    # Calculate intersection area
    x0 = max(bbox_a["x"], bbox_b["x"])
    y0 = max(bbox_a["y"], bbox_b["y"])
    x1 = min(bbox_a["x"] + bbox_a["width"], bbox_b["x"] + bbox_b["width"])
    y1 = min(bbox_a["y"] + bbox_a["height"], bbox_b["y"] + bbox_b["height"])
    
    if x1 <= x0 or y1 <= y0:
        return 0.0
    
    intersection = (x1 - x0) * (y1 - y0)
    area_a = bbox_a["width"] * bbox_a["height"]
    area_b = bbox_b["width"] * bbox_b["height"]
    union = area_a + area_b - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def _compare_caption_pair(
    caption_a: FigureCaption,
    caption_b: FigureCaption,
    page_num: int,
    confidence: float,
    page_width: float,
    page_height: float,
) -> List[Diff]:
    """Compare two matched figure captions."""
    from comparison.models import Diff
    
    diffs = []
    
    normalized_bbox = normalize_bbox(
        (caption_a.figure_bbox["x"], caption_a.figure_bbox["y"],
         caption_a.figure_bbox["x"] + caption_a.figure_bbox["width"],
         caption_a.figure_bbox["y"] + caption_a.figure_bbox["height"]),
        page_width, page_height
    )
    
    # Check for numbering change
    if caption_a.caption_number is not None and caption_b.caption_number is not None:
        if caption_a.caption_number != caption_b.caption_number:
            diffs.append(Diff(
                page_num=page_num,
                diff_type="modified",
                change_type="formatting",
                old_text=caption_a.caption_text or f"Figure {caption_a.caption_number}",
                new_text=caption_b.caption_text or f"Figure {caption_b.caption_number}",
                bbox=normalized_bbox,
                confidence=confidence,
                metadata={
                    "figure_change": "numbering",
                    "old_number": caption_a.caption_number,
                    "new_number": caption_b.caption_number,
                    "page_width": page_width,
                    "page_height": page_height,
                },
            ))
    
    # Check for caption text change (excluding numbering)
    text_a_clean = _strip_caption_number(caption_a.caption_text)
    text_b_clean = _strip_caption_number(caption_b.caption_text)
    
    # Use normalized comparison to ignore case and minor differences
    if normalize_text(text_a_clean) != normalize_text(text_b_clean):
        diffs.append(Diff(
            page_num=page_num,
            diff_type="modified",
            change_type="content",
            old_text=caption_a.caption_text,
            new_text=caption_b.caption_text,
            bbox=normalized_bbox if caption_a.caption_bbox is None else None,
            confidence=confidence,
            metadata={
                "figure_change": "caption_text",
                "page_width": page_width,
                "page_height": page_height,
            },
        ))
    
    return diffs


def _strip_caption_number(caption_text: str) -> str:
    """Remove figure number from caption text for comparison."""
    if not caption_text:
        return ""
    
    # Remove patterns like "Figure 1:", "Fig. 2.", etc.
    text = re.sub(r'^(figure|fig\.?)\s+\d+(?:[-.]\d+)?\s*[:.]?\s*', '', caption_text, flags=re.IGNORECASE)
    return text.strip()

