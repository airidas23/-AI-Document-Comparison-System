"""Document alignment utilities.

Phase 2 Optimization: Candidate Generation (Step 2)
- Length-ratio prefilter to skip obviously mismatched pairs
- N-gram hash check for quick similarity estimation  
- BBox proximity filter to reduce search space
- Statistics tracking for skipped candidates
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

from rapidfuzz import fuzz

from comparison.hierarchical_alignment import hierarchical_align, segment_document
from comparison.models import PageData, TextBlock
from utils.logging import logger
from utils.text_normalization import normalize_text


AlignmentMap = Dict[int, Tuple[int, float]]  # page_a -> (page_b, confidence)


# =============================================================================
# Phase 2: Candidate Generation Statistics
# =============================================================================

@dataclass
class CandidateGenerationStats:
    """Statistics for candidate generation optimization (Phase 2 - Step 2)."""
    total_pairs_considered: int = 0
    pairs_skipped_length_ratio: int = 0
    pairs_skipped_ngram_hash: int = 0
    pairs_skipped_bbox_distance: int = 0
    pairs_after_prefilter: int = 0
    fallback_relaxations: int = 0
    
    def reset(self) -> None:
        """Reset statistics for a new comparison."""
        self.total_pairs_considered = 0
        self.pairs_skipped_length_ratio = 0
        self.pairs_skipped_ngram_hash = 0
        self.pairs_skipped_bbox_distance = 0
        self.pairs_after_prefilter = 0
        self.fallback_relaxations = 0
    
    @property
    def skip_ratio(self) -> float:
        """Ratio of pairs skipped by prefilters."""
        if self.total_pairs_considered == 0:
            return 0.0
        skipped = (
            self.pairs_skipped_length_ratio 
            + self.pairs_skipped_ngram_hash 
            + self.pairs_skipped_bbox_distance
        )
        return skipped / self.total_pairs_considered
    
    def to_dict(self) -> dict:
        """Export to JSON-serializable dict."""
        return {
            "total_pairs_considered": self.total_pairs_considered,
            "pairs_skipped_length_ratio": self.pairs_skipped_length_ratio,
            "pairs_skipped_ngram_hash": self.pairs_skipped_ngram_hash,
            "pairs_skipped_bbox_distance": self.pairs_skipped_bbox_distance,
            "pairs_after_prefilter": self.pairs_after_prefilter,
            "fallback_relaxations": self.fallback_relaxations,
            "skip_ratio": self.skip_ratio,
        }


# Global stats instance for current comparison
_candidate_stats = CandidateGenerationStats()


def get_candidate_stats() -> CandidateGenerationStats:
    """Get current candidate generation statistics."""
    return _candidate_stats


def reset_candidate_stats() -> None:
    """Reset candidate generation statistics."""
    _candidate_stats.reset()


# =============================================================================
# Phase 2: Prefilter Functions
# =============================================================================

def _compute_ngram_hash(text: str, n: int = 3) -> Set[int]:
    """Compute set of n-gram hashes for quick similarity estimation.
    
    Uses character n-grams for robustness to OCR noise.
    """
    if not text or len(text) < n:
        return set()
    
    # Normalize and lowercase for comparison
    text = text.lower().strip()
    hashes = set()
    for i in range(len(text) - n + 1):
        ngram = text[i:i+n]
        hashes.add(hash(ngram) & 0xFFFF)  # 16-bit hash for memory efficiency
    return hashes


def _ngram_similarity(hashes_a: Set[int], hashes_b: Set[int]) -> float:
    """Jaccard similarity of n-gram hash sets."""
    if not hashes_a or not hashes_b:
        return 0.0
    intersection = len(hashes_a & hashes_b)
    union = len(hashes_a | hashes_b)
    return intersection / union if union > 0 else 0.0


def _length_ratio_filter(text_a: str, text_b: str, max_ratio: float = 3.0) -> bool:
    """Check if texts have compatible lengths.
    
    Returns True if texts are compatible, False if should skip.
    """
    len_a = len(text_a.strip()) if text_a else 0
    len_b = len(text_b.strip()) if text_b else 0
    
    if len_a == 0 or len_b == 0:
        return len_a == len_b  # Both empty is compatible
    
    ratio = max(len_a, len_b) / min(len_a, len_b)
    return ratio <= max_ratio


def _bbox_distance(bbox_a: dict, bbox_b: dict) -> float:
    """Calculate center-to-center distance between bounding boxes."""
    cx_a = bbox_a.get("x", 0) + bbox_a.get("width", 0) / 2
    cy_a = bbox_a.get("y", 0) + bbox_a.get("height", 0) / 2
    cx_b = bbox_b.get("x", 0) + bbox_b.get("width", 0) / 2
    cy_b = bbox_b.get("y", 0) + bbox_b.get("height", 0) / 2
    
    return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5


def prefilter_candidate_pair(
    text_a: str,
    text_b: str,
    bbox_a: Optional[dict] = None,
    bbox_b: Optional[dict] = None,
    *,
    length_ratio_max: float = 3.0,
    ngram_threshold: float = 0.1,
    bbox_distance_max: float = 500.0,
    use_ngram: bool = True,
    use_bbox: bool = True,
    is_ocr: bool = False,
) -> Tuple[bool, str]:
    """Apply Phase 2 prefilters to candidate pair.
    
    Returns:
        Tuple of (should_compare: bool, skip_reason: str)
        If should_compare is False, skip_reason explains why.
    """
    global _candidate_stats
    _candidate_stats.total_pairs_considered += 1
    
    # Relax filters for OCR (more noise tolerance)
    if is_ocr:
        length_ratio_max *= 1.5
        ngram_threshold *= 0.5
        bbox_distance_max *= 1.5
    
    # Filter 1: Length ratio
    if not _length_ratio_filter(text_a, text_b, length_ratio_max):
        _candidate_stats.pairs_skipped_length_ratio += 1
        return (False, "length_ratio")
    
    # Filter 2: N-gram hash similarity
    if use_ngram and text_a and text_b:
        hashes_a = _compute_ngram_hash(text_a)
        hashes_b = _compute_ngram_hash(text_b)
        
        if hashes_a and hashes_b:
            ngram_sim = _ngram_similarity(hashes_a, hashes_b)
            if ngram_sim < ngram_threshold:
                _candidate_stats.pairs_skipped_ngram_hash += 1
                return (False, "ngram_hash")
    
    # Filter 3: BBox distance (if available)
    if use_bbox and bbox_a and bbox_b:
        dist = _bbox_distance(bbox_a, bbox_b)
        if dist > bbox_distance_max:
            _candidate_stats.pairs_skipped_bbox_distance += 1
            return (False, "bbox_distance")
    
    _candidate_stats.pairs_after_prefilter += 1
    return (True, "")


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

    extraction_method_a = (page_a.metadata or {}).get("extraction_method") or (page_a.metadata or {}).get("line_extraction_method", "")
    is_ocr_a = "ocr" in str(extraction_method_a or "").lower()
    
    # Simple text overlap similarity using normalized text
    normalized_a = normalize_text(text_a, ocr=is_ocr_a)
    words_a = set(normalized_a.split())
    
    for idx in range(start_idx, end_idx):
        if idx >= len(doc_b):
            break
        page_b = doc_b[idx]
        text_b = " ".join(block.text for block in page_b.blocks if block.text.strip())
        extraction_method_b = (page_b.metadata or {}).get("extraction_method") or (page_b.metadata or {}).get("line_extraction_method", "")
        is_ocr_b = "ocr" in str(extraction_method_b or "").lower()
        normalized_b = normalize_text(text_b, ocr=(is_ocr_a or is_ocr_b))
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
    
    # Fallback: OCR-aware alignment when needed, otherwise simple positional.
    from config.settings import settings

    extraction_method_a = (page_a.metadata or {}).get("extraction_method", "")
    extraction_method_b = (page_b.metadata or {}).get("extraction_method", "")
    is_ocr = ("ocr" in (extraction_method_a or "").lower()) or ("ocr" in (extraction_method_b or "").lower())

    dx, dy, translation_conf = (0.0, 0.0, 0.0)
    if is_ocr:
        dx, dy, translation_conf = _estimate_page_translation(
            page_a,
            page_b,
            min_similarity=settings.ocr_translation_estimation_min_similarity,
        )
        page_a.metadata["page_alignment_translation"] = {
            "dx": float(dx),
            "dy": float(dy),
            "confidence": float(translation_conf),
            "method": "median_block_centers",
        }

    mapping: Dict[int, int] = {}
    used_b: set[int] = set()

    def _center(b: TextBlock) -> tuple[float, float]:
        return (
            b.bbox["x"] + b.bbox["width"] / 2,
            b.bbox["y"] + b.bbox["height"] / 2,
        )

    def _block_text_similarity(text_a: str, text_b: str) -> float:
        if not text_a and not text_b:
            return 1.0
        if not text_a or not text_b:
            return 0.0
        na = normalize_text(text_a, ocr=is_ocr)
        nb = normalize_text(text_b, ocr=is_ocr)
        if not na or not nb:
            return 0.0
        return fuzz.token_sort_ratio(na, nb) / 100.0

    # Match longer / more informative blocks first to reduce collisions.
    block_indices_a = list(range(len(page_a.blocks)))
    block_indices_a.sort(key=lambda i: len((page_a.blocks[i].text or "").strip()), reverse=True)

    for idx_a in block_indices_a:
        block_a = page_a.blocks[idx_a]
        ax, ay = _center(block_a)
        text_a = block_a.text or ""

        best_idx_b: Optional[int] = None
        best_score = 0.0

        for idx_b, block_b in enumerate(page_b.blocks):
            if idx_b in used_b:
                continue
            
            text_b = block_b.text or ""
            
            # Phase 2: Apply prefilters before expensive similarity computation
            should_compare, skip_reason = prefilter_candidate_pair(
                text_a,
                text_b,
                bbox_a=block_a.bbox,
                bbox_b=block_b.bbox,
                length_ratio_max=3.0,
                ngram_threshold=0.05,  # Low threshold since we do proper similarity later
                bbox_distance_max=settings.block_alignment_distance_threshold,
                use_ngram=len(text_a) > 20 and len(text_b) > 20,  # Only for substantial text
                use_bbox=True,
                is_ocr=is_ocr,
            )
            
            if not should_compare:
                continue

            bx, by = _center(block_b)
            if is_ocr:
                bx -= dx
                by -= dy

            distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            if distance > settings.block_alignment_distance_threshold:
                continue

            if is_ocr:
                text_score = _block_text_similarity(text_a, text_b)
                if text_score < settings.ocr_min_text_similarity_for_match:
                    continue
                # OCR: text must dominate; geometry stabilizes ties.
                position_score = max(0.0, 1.0 - min(distance / max(settings.block_alignment_distance_threshold, 1.0), 1.0))
                score = (0.8 * text_score) + (0.2 * position_score)
            else:
                # Non-OCR: text dominates; geometry stabilizes.
                # This prevents misalignment of similar-looking blocks (e.g., same-width paragraphs).
                text_score = _block_text_similarity(text_a, text_b)
                if text_score < 0.30:
                    continue
                position_score = max(0.0, 1.0 - min(distance / max(settings.block_alignment_distance_threshold, 1.0), 1.0))
                score = (0.7 * text_score) + (0.3 * position_score)

            if score > best_score:
                best_score = score
                best_idx_b = idx_b

        if best_idx_b is not None:
            mapping[idx_a] = best_idx_b
            used_b.add(best_idx_b)
    
    # Phase 2: Log prefilter statistics
    stats = get_candidate_stats()
    if stats.total_pairs_considered > 0:
        logger.debug(
            "Candidate generation: %d pairs, %.1f%% skipped (length=%d, ngram=%d, bbox=%d)",
            stats.total_pairs_considered,
            stats.skip_ratio * 100,
            stats.pairs_skipped_length_ratio,
            stats.pairs_skipped_ngram_hash,
            stats.pairs_skipped_bbox_distance,
        )

    return mapping


def _estimate_page_translation(
    page_a: PageData,
    page_b: PageData,
    *,
    min_similarity: float = 0.85,
    max_pairs: int = 200,
) -> tuple[float, float, float]:
    """Estimate a global (dx, dy) translation between pages using high-similarity blocks.

    Returns (dx, dy, confidence) where dx/dy are in the same coordinate space as bboxes.
    """
    pairs: List[tuple[float, float, float]] = []  # (sim, dx, dy)

    def _center(b: TextBlock) -> tuple[float, float]:
        return (
            b.bbox["x"] + b.bbox["width"] / 2,
            b.bbox["y"] + b.bbox["height"] / 2,
        )

    blocks_a = [b for b in page_a.blocks if (b.text or "").strip()]
    blocks_b = [b for b in page_b.blocks if (b.text or "").strip()]
    if not blocks_a or not blocks_b:
        return (0.0, 0.0, 0.0)

    # Prefer informative blocks for translation anchors.
    blocks_a = sorted(blocks_a, key=lambda b: len((b.text or "").strip()), reverse=True)[:80]
    blocks_b = sorted(blocks_b, key=lambda b: len((b.text or "").strip()), reverse=True)[:120]

    for ba in blocks_a:
        na = normalize_text(ba.text, ocr=True)
        if not na:
            continue
        ax, ay = _center(ba)

        for bb in blocks_b:
            nb = normalize_text(bb.text, ocr=True)
            if not nb:
                continue
            sim = fuzz.token_sort_ratio(na, nb) / 100.0
            if sim < min_similarity:
                continue
            bx, by = _center(bb)
            pairs.append((sim, bx - ax, by - ay))
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break

    if len(pairs) < 3:
        return (0.0, 0.0, 0.0)

    pairs.sort(key=lambda t: t[0], reverse=True)
    top = pairs[: min(len(pairs), 50)]
    dxs = sorted(d for _, d, _ in top)
    dys = sorted(d for _, _, d in top)
    mid = len(top) // 2
    dx = dxs[mid]
    dy = dys[mid]

    # Confidence: combine count and median similarity.
    sims = sorted(s for s, _, _ in top)
    median_sim = sims[len(sims) // 2]
    confidence = min(1.0, (len(top) / 20.0) * median_sim)
    return (float(dx), float(dy), float(confidence))


def detect_layout_shift(
    block_a: TextBlock,
    block_b: TextBlock,
    page_width: float,
    page_height: float,
    tolerance_ratio: float = 0.01,
    translation: Optional[Dict[str, float]] = None,
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
    
    tx = float((translation or {}).get("dx", 0.0))
    ty = float((translation or {}).get("dy", 0.0))

    # Calculate position differences (optionally compensate global translation)
    dx = abs((bbox_b["x"] - tx) - bbox_a["x"])
    dy = abs((bbox_b["y"] - ty) - bbox_a["y"])
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
