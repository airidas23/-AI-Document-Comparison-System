"""Figure caption and numbering comparison with perceptual hashing."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from comparison.alignment import align_pages
from comparison.models import Diff, PageData, FigureRegion
from utils.coordinates import normalize_bbox
from utils.logging import logger
from utils.text_normalization import normalize_text

# Optional: imagehash for perceptual hashing
try:
    import imagehash
    from PIL import Image
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    logger.warning("imagehash not available - figure visual comparison disabled")


@dataclass
class FigureCaption:
    """Represents a figure with its caption."""
    figure_bbox: Dict[str, float]
    caption_text: str
    page_num: int
    caption_number: Optional[int] = None
    caption_label: Optional[str] = None  # "Figure", "Fig.", etc.
    caption_bbox: Optional[Dict[str, float]] = None
    image_data: Optional[bytes] = None  # Raw image bytes for visual comparison
    phash: Optional[str] = None  # Perceptual hash string
    dhash: Optional[str] = None  # Difference hash string


def compute_figure_hashes(image_data: bytes) -> Tuple[Optional[str], Optional[str]]:
    """
    Compute perceptual and difference hashes for a figure image.
    
    Args:
        image_data: Raw image bytes (PNG, JPEG, etc.)
    
    Returns:
        Tuple of (phash_string, dhash_string), or (None, None) if unavailable
    """
    if not IMAGEHASH_AVAILABLE or not image_data:
        return None, None
    
    try:
        img = Image.open(io.BytesIO(image_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Perceptual hash - good for similar images with minor edits
        phash = str(imagehash.phash(img))
        
        # Difference hash - good for structural changes
        dhash = str(imagehash.dhash(img))
        
        return phash, dhash
    except Exception as e:
        logger.warning("Failed to compute image hashes: %s", e)
        return None, None


def compare_figure_hashes(
    phash_a: Optional[str],
    phash_b: Optional[str],
    dhash_a: Optional[str],
    dhash_b: Optional[str],
    threshold: int = 8,
) -> Tuple[float, str]:
    """
    Compare two figures using their perceptual hashes.
    
    Args:
        phash_a: Perceptual hash of first figure
        phash_b: Perceptual hash of second figure
        dhash_a: Difference hash of first figure
        dhash_b: Difference hash of second figure
        threshold: Hamming distance threshold (default 8, max 64)
    
    Returns:
        Tuple of (similarity_score 0-1, change_type)
        - change_type: "identical", "similar", "different"
    """
    if not IMAGEHASH_AVAILABLE:
        return 1.0, "unknown"
    
    if not all([phash_a, phash_b]):
        return 0.5, "unknown"
    
    try:
        # Convert string hashes to imagehash objects
        hash_a_p = imagehash.hex_to_hash(phash_a)
        hash_b_p = imagehash.hex_to_hash(phash_b)
        
        # Hamming distance (0 = identical, 64 = completely different)
        phash_distance = hash_a_p - hash_b_p
        
        # Also check dhash for structural changes
        dhash_distance = 64  # Default to max if not available
        if dhash_a and dhash_b:
            hash_a_d = imagehash.hex_to_hash(dhash_a)
            hash_b_d = imagehash.hex_to_hash(dhash_b)
            dhash_distance = hash_a_d - hash_b_d
        
        # Use the more conservative (larger) distance
        max_distance = max(phash_distance, dhash_distance)
        
        # Convert to similarity score (0-1)
        similarity = 1.0 - (max_distance / 64.0)
        
        # Categorize the change
        if max_distance == 0:
            change_type = "identical"
        elif max_distance <= threshold:
            change_type = "similar"
        else:
            change_type = "different"
        
        return similarity, change_type
    
    except Exception as e:
        logger.warning("Failed to compare hashes: %s", e)
        return 0.5, "unknown"


def extract_figure_image(page_data: Any, figure_bbox: Dict[str, float]) -> Optional[bytes]:
    """
    Extract figure image bytes from a PDF page.
    
    This requires access to the original PDF document or rendered page.
    
    Args:
        page_data: Page data or fitz.Page object
        figure_bbox: Figure bounding box
    
    Returns:
        Raw image bytes or None
    """
    # Check if we have a fitz page reference
    if hasattr(page_data, 'fitz_page') and page_data.fitz_page is not None:
        try:
            import fitz
            page = page_data.fitz_page
            
            # Convert bbox to fitz.Rect
            rect = fitz.Rect(
                figure_bbox["x"],
                figure_bbox["y"],
                figure_bbox["x"] + figure_bbox["width"],
                figure_bbox["y"] + figure_bbox["height"]
            )
            
            # Render at 2x resolution for better quality
            mat = fitz.Matrix(2, 2)
            clip = rect
            
            pix = page.get_pixmap(matrix=mat, clip=clip)
            return pix.tobytes("png")
        
        except Exception as e:
            logger.debug("Could not extract figure image: %s", e)

    # Fallback: reopen the source PDF (if available) and render the clipped region.
    # This is the common path for PageData objects that don't carry a live fitz.Page.
    try:
        source_path = None
        page_index = None
        if hasattr(page_data, "metadata") and isinstance(page_data.metadata, dict):
            source_path = page_data.metadata.get("source_pdf_path")
            page_index = page_data.metadata.get("page_index")

        if source_path:
            import fitz

            # Default to 0-based page index derived from 1-based page_num.
            if page_index is None and hasattr(page_data, "page_num"):
                try:
                    page_index = int(page_data.page_num) - 1
                except Exception:
                    page_index = None

            with fitz.open(source_path) as doc:
                if page_index is None or page_index < 0 or page_index >= len(doc):
                    return None
                page = doc[page_index]
                rect = fitz.Rect(
                    figure_bbox["x"],
                    figure_bbox["y"],
                    figure_bbox["x"] + figure_bbox["width"],
                    figure_bbox["y"] + figure_bbox["height"],
                )
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat, clip=rect)
                return pix.tobytes("png")
    except Exception as e:
        logger.debug("Could not extract figure image via source_pdf_path: %s", e)
    
    # Check if image data is in metadata
    if hasattr(page_data, 'metadata') and page_data.metadata:
        figures = page_data.metadata.get("figures", [])
        for fig in figures:
            fig_bbox = fig.get("bbox")
            if not fig_bbox:
                continue
            
            # Handle both list and dict bbox formats
            if isinstance(fig_bbox, dict):
                fig_x = fig_bbox.get("x", 0)
                fig_y = fig_bbox.get("y", 0)
            elif isinstance(fig_bbox, (list, tuple)) and len(fig_bbox) >= 2:
                fig_x = fig_bbox[0]
                fig_y = fig_bbox[1]
            else:
                continue
            
            # Check if this is the matching figure
            if (abs(fig_x - figure_bbox["x"]) < 5 and
                abs(fig_y - figure_bbox["y"]) < 5):
                return fig.get("image_data")
    
    return None


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
                page_a.width, page_a.height,
                page_a=page_a, page_b=page_b  # Pass pages for visual comparison
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
                    "type": "figure",
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
                    "type": "figure",
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
    page_a: Optional[PageData] = None,
    page_b: Optional[PageData] = None,
) -> List[Diff]:
    """Compare two matched figure captions, including visual comparison."""
    from comparison.models import Diff
    from config.settings import settings
    
    diffs = []
    
    normalized_bbox = normalize_bbox(
        (caption_a.figure_bbox["x"], caption_a.figure_bbox["y"],
         caption_a.figure_bbox["x"] + caption_a.figure_bbox["width"],
         caption_a.figure_bbox["y"] + caption_a.figure_bbox["height"]),
        page_width, page_height
    )
    
    # Visual comparison using perceptual hashing
    if IMAGEHASH_AVAILABLE and page_a and page_b:
        # Extract images if not already present
        if caption_a.image_data is None:
            caption_a.image_data = extract_figure_image(page_a, caption_a.figure_bbox)
        if caption_b.image_data is None:
            caption_b.image_data = extract_figure_image(page_b, caption_b.figure_bbox)
        
        # Compute hashes if we have images
        if caption_a.image_data and caption_b.image_data:
            if caption_a.phash is None:
                caption_a.phash, caption_a.dhash = compute_figure_hashes(caption_a.image_data)
            if caption_b.phash is None:
                caption_b.phash, caption_b.dhash = compute_figure_hashes(caption_b.image_data)
            
            # Compare hashes
            threshold = getattr(settings, 'figure_hash_threshold', 8)
            similarity, visual_change = compare_figure_hashes(
                caption_a.phash, caption_b.phash,
                caption_a.dhash, caption_b.dhash,
                threshold=threshold
            )
            
            if visual_change == "different":
                diffs.append(Diff(
                    page_num=page_num,
                    diff_type="modified",
                    change_type="visual",
                    old_text=caption_a.caption_text or "Figure (visual)",
                    new_text=caption_b.caption_text or "Figure (visual)",
                    bbox=normalized_bbox,
                    confidence=similarity,
                    metadata={
                        "type": "figure",
                        "figure_change": "visual_content",
                        "phash_a": caption_a.phash,
                        "phash_b": caption_b.phash,
                        "dhash_a": caption_a.dhash,
                        "dhash_b": caption_b.dhash,
                        "visual_similarity": similarity,
                        "page_width": page_width,
                        "page_height": page_height,
                    },
                ))
    
    # Check for numbering change
    if caption_a.caption_number is not None and caption_b.caption_number is not None:
        if caption_a.caption_number != caption_b.caption_number:
            diffs.append(Diff(
                page_num=page_num,
                diff_type="modified",
                change_type="visual",
                old_text=caption_a.caption_text or f"Figure {caption_a.caption_number}",
                new_text=caption_b.caption_text or f"Figure {caption_b.caption_number}",
                bbox=normalized_bbox,
                confidence=confidence,
                metadata={
                    "type": "figure",
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
                "type": "figure",
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

