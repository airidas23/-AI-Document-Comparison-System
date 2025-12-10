"""Document alignment utilities."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from comparison.hierarchical_alignment import hierarchical_align, segment_document
from comparison.models import PageData, TextBlock
from utils.logging import logger
from utils.text_normalization import normalize_text


AlignmentMap = Dict[int, Tuple[int, float]]  # page_a -> (page_b, confidence)


def align_pages(
    doc_a: List[PageData],
    doc_b: List[PageData],
    use_similarity: bool = True,
) -> AlignmentMap:
    """
    Align pages between two documents using position and text similarity.
    
    Args:
        doc_a: First document pages
        doc_b: Second document pages
        use_similarity: If True, use text similarity for better alignment
    
    Returns:
        Mapping from doc_a page numbers to (doc_b page number, confidence)
    """
    logger.info("Aligning %d pages -> %d pages", len(doc_a), len(doc_b))
    mapping: AlignmentMap = {}
    
    if not doc_a or not doc_b:
        return mapping
    
    # Simple positional alignment first
    for idx, page_a in enumerate(doc_a):
        # Default: align by position
        target_idx = min(idx, len(doc_b) - 1)
        page_b = doc_b[target_idx]
        confidence = 1.0 if len(doc_a) == len(doc_b) else 0.8
        
        # If similarity matching is enabled, try to find better match
        if use_similarity and len(doc_a) != len(doc_b):
            best_match_idx, best_confidence = _find_best_page_match(
                page_a, doc_b, start_idx=max(0, idx - 2), end_idx=min(len(doc_b), idx + 3)
            )
            if best_confidence > confidence:
                target_idx = best_match_idx
                confidence = best_confidence
        
        mapping[page_a.page_num] = (doc_b[target_idx].page_num, confidence)
    
    logger.debug("Alignment complete: %d mappings", len(mapping))
    return mapping


def _find_best_page_match(
    page_a: PageData,
    doc_b: List[PageData],
    start_idx: int = 0,
    end_idx: int | None = None,
) -> Tuple[int, float]:
    """Find the best matching page in doc_b for page_a using text similarity."""
    if end_idx is None:
        end_idx = len(doc_b)
    
    end_idx = min(end_idx, len(doc_b))
    
    # Extract text from page_a
    text_a = " ".join(block.text for block in page_a.blocks if block.text.strip())
    if not text_a:
        # Fallback to positional alignment
        return (min(start_idx, len(doc_b) - 1), 0.5)
    
    best_idx = start_idx
    best_similarity = 0.0
    
    # Simple text overlap similarity using normalized text
    normalized_a = normalize_text(text_a)
    words_a = set(normalized_a.split())
    
    for idx in range(start_idx, end_idx):
        if idx >= len(doc_b):
            break
        page_b = doc_b[idx]
        text_b = " ".join(block.text for block in page_b.blocks if block.text.strip())
        normalized_b = normalize_text(text_b)
        words_b = set(normalized_b.split())
        
        if not words_b:
            continue
        
        # Jaccard similarity
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        similarity = intersection / union if union > 0 else 0.0
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_idx = idx
    
    # Convert similarity to confidence (0.5 to 1.0 range)
    confidence = 0.5 + (best_similarity * 0.5)
    return (best_idx, confidence)


def align_sections(
    page_a: PageData,
    page_b: PageData,
    use_hierarchical: bool = True,
) -> Dict[int, int]:
    """
    Align text blocks between two pages.
    
    If Markdown structure is available, uses hierarchical alignment.
    Otherwise falls back to positional alignment.
    
    Args:
        page_a: First page
        page_b: Second page
        use_hierarchical: If True, use hierarchical alignment when available
    
    Returns:
        Mapping from block index in page_a to block index in page_b
    """
    if not page_a.blocks or not page_b.blocks:
        return {}
    
    # Check if hierarchical alignment is available and should be used
    has_markdown_a = page_a.metadata.get("has_markdown_structure", False)
    has_markdown_b = page_b.metadata.get("has_markdown_structure", False)
    
    if use_hierarchical and (has_markdown_a or has_markdown_b):
        try:
            # Use hierarchical alignment
            segment_alignment = hierarchical_align(page_a, page_b, use_dtw=True)
            
            # Convert segment alignment to block alignment
            segments_a = segment_document(page_a)
            segments_b = segment_document(page_b)
            
            block_mapping: Dict[int, int] = {}
            
            for seg_idx_a, (seg_idx_b, confidence) in segment_alignment.items():
                if seg_idx_a >= len(segments_a) or seg_idx_b >= len(segments_b):
                    continue
                
                seg_a = segments_a[seg_idx_a]
                seg_b = segments_b[seg_idx_b]
                
                # Map blocks within segments
                for block_idx_a, block_a in enumerate(seg_a.blocks):
                    if block_idx_a < len(seg_b.blocks):
                        # Find block index in original page
                        try:
                            block_idx_in_page_a = page_a.blocks.index(block_a)
                            block_idx_in_page_b = page_b.blocks.index(seg_b.blocks[block_idx_a])
                            block_mapping[block_idx_in_page_a] = block_idx_in_page_b
                        except (ValueError, IndexError):
                            # Block not found, skip
                            continue
            
            if block_mapping:
                logger.debug("Used hierarchical alignment: %d block mappings", len(block_mapping))
                return block_mapping
        except Exception as exc:
            logger.warning("Hierarchical alignment failed, falling back to positional: %s", exc)
    
    # Fallback to simple positional alignment
    mapping: Dict[int, int] = {}
    
    for idx_a, block_a in enumerate(page_a.blocks):
        # Find closest block in page_b by position
        best_idx_b = 0
        min_distance = float("inf")
        
        for idx_b, block_b in enumerate(page_b.blocks):
            # Calculate distance between block centers
            # bbox is in dict format: {"x": x, "y": y, "width": w, "height": h}
            ax = block_a.bbox["x"] + block_a.bbox["width"] / 2
            ay = block_a.bbox["y"] + block_a.bbox["height"] / 2
            bx = block_b.bbox["x"] + block_b.bbox["width"] / 2
            by = block_b.bbox["y"] + block_b.bbox["height"] / 2
            
            distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                best_idx_b = idx_b
        
        from config.settings import settings
        if min_distance < settings.block_alignment_distance_threshold:
            mapping[idx_a] = best_idx_b
    
    return mapping


def detect_layout_shift(
    block_a: TextBlock,
    block_b: TextBlock,
    page_width: float,
    page_height: float,
    tolerance_ratio: float = 0.01,
) -> Optional[Dict[str, float]]:
    """
    Detect layout shift between two aligned blocks.
    
    A layout shift is detected if the block position has changed by more than
    the tolerance ratio (default 1% of page dimensions).
    
    Args:
        block_a: Block from first document
        block_b: Block from second document
        page_width: Page width for calculating relative shift
        page_height: Page height for calculating relative shift
        tolerance_ratio: Relative tolerance (0.01 = 1% of page dimension)
    
    Returns:
        Dict with shift information if shift detected, None otherwise
    """
    bbox_a = block_a.bbox
    bbox_b = block_b.bbox
    
    # Calculate position differences
    dx = abs(bbox_b["x"] - bbox_a["x"])
    dy = abs(bbox_b["y"] - bbox_a["y"])
    dw = abs(bbox_b["width"] - bbox_a["width"])
    dh = abs(bbox_b["height"] - bbox_a["height"])
    
    # Calculate relative shifts
    rel_dx = dx / page_width if page_width > 0 else 0.0
    rel_dy = dy / page_height if page_height > 0 else 0.0
    rel_dw = dw / page_width if page_width > 0 else 0.0
    rel_dh = dh / page_height if page_height > 0 else 0.0
    
    # Check if any shift exceeds tolerance
    if (rel_dx > tolerance_ratio or rel_dy > tolerance_ratio or
        rel_dw > tolerance_ratio or rel_dh > tolerance_ratio):
        return {
            "shift_detected": True,
            "dx": dx,
            "dy": dy,
            "dw": dw,
            "dh": dh,
            "rel_dx": rel_dx,
            "rel_dy": rel_dy,
            "rel_dw": rel_dw,
            "rel_dh": rel_dh,
            "tolerance_ratio": tolerance_ratio,
        }
    
    return None
