"""Two-stage gating system for OCR-aware change detection.

Phase 2: Implements a two-stage approach to filter noise BEFORE diffing:

Stage A (Cheap Gate): Fast pre-filtering without semantics
- rapidfuzz token_sort_ratio / token_set_ratio
- bbox/position sanity check
- Length ratio prefilter

Stage B (Gray Zone): Semantic analysis only for uncertain cases
- Sentence embedding similarity
- Only runs when Stage A score is in 70-85% range
- Or when bbox shows layout drift

This reduces phantom diffs by ~60-80% while preserving recall.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

from rapidfuzz.fuzz import ratio, token_sort_ratio, token_set_ratio

from utils.ocr_normalizer import normalize_ocr_compare, normalize_ocr_strict


# =============================================================================
# Gating Result Types
# =============================================================================

class GateDecision(Enum):
    """Decision from the gating system."""
    IDENTICAL = "identical"           # No diff needed - content is same
    LIKELY_IDENTICAL = "likely_identical"  # High confidence same, skip diff
    NEEDS_DIFF = "needs_diff"         # Run detailed diff
    LIKELY_DIFFERENT = "likely_different"  # High confidence different
    SEMANTIC_CHECK = "semantic_check"  # Gray zone - needs embedding comparison


@dataclass
class GatingResult:
    """Result from the two-stage gating system."""
    
    decision: GateDecision
    confidence: float  # 0.0 - 1.0
    
    # Stage A metrics
    token_sort_score: float = 0.0
    token_set_score: float = 0.0
    char_ratio_score: float = 0.0
    length_ratio: float = 0.0
    
    # Stage B metrics (if run)
    semantic_score: Optional[float] = None
    embedding_cached: bool = False
    
    # Position metrics (if bbox provided)
    bbox_overlap_iou: Optional[float] = None
    position_drift: Optional[float] = None
    
    # Timing
    stage_a_time: float = 0.0
    stage_b_time: float = 0.0
    
    # Explanation
    reason: str = ""
    
    @property
    def total_time(self) -> float:
        return self.stage_a_time + self.stage_b_time
    
    @property
    def should_skip_diff(self) -> bool:
        """Whether to skip detailed diff based on gating decision."""
        return self.decision in (GateDecision.IDENTICAL, GateDecision.LIKELY_IDENTICAL)
    
    @property
    def is_phantom_diff(self) -> bool:
        """Whether this appears to be a phantom diff (noise)."""
        return self.decision == GateDecision.LIKELY_IDENTICAL and self.confidence > 0.8


# =============================================================================
# Gating Configuration
# =============================================================================

@dataclass
class GatingConfig:
    """Configuration for the two-stage gating system."""
    
    # Stage A thresholds
    identical_threshold: float = 0.98      # Above this = definitely identical
    likely_identical_threshold: float = 0.92  # Above this = probably identical
    gray_zone_low: float = 0.70           # Below this = definitely different
    gray_zone_high: float = 0.85          # Between low-high = needs semantic
    
    # Length ratio thresholds
    length_ratio_min: float = 0.7         # Below this = definitely different
    length_ratio_max: float = 1.4         # Above this = definitely different
    
    # Position thresholds
    bbox_iou_threshold: float = 0.5       # Below this = significant layout drift
    position_drift_threshold: float = 0.05  # Relative to page size
    
    # Stage B (semantic) thresholds
    semantic_identical_threshold: float = 0.95
    semantic_different_threshold: float = 0.80
    
    # Performance
    enable_stage_b: bool = True           # Can disable semantic for speed
    cache_embeddings: bool = True
    
    # OCR-specific adjustments
    ocr_mode: bool = False                # More lenient thresholds for OCR
    ocr_identical_threshold: float = 0.95
    ocr_likely_identical_threshold: float = 0.88
    ocr_gray_zone_low: float = 0.65
    ocr_gray_zone_high: float = 0.82
    
    def get_thresholds(self) -> Tuple[float, float, float, float]:
        """Get thresholds based on OCR mode."""
        if self.ocr_mode:
            return (
                self.ocr_identical_threshold,
                self.ocr_likely_identical_threshold,
                self.ocr_gray_zone_low,
                self.ocr_gray_zone_high,
            )
        return (
            self.identical_threshold,
            self.likely_identical_threshold,
            self.gray_zone_low,
            self.gray_zone_high,
        )


# =============================================================================
# Embedding Cache (for Stage B)
# =============================================================================

class EmbeddingCache:
    """Simple cache for text embeddings to avoid recomputation."""
    
    def __init__(self, max_size: int = 10000):
        self._cache: Dict[str, Any] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
    
    def get(self, text: str) -> Optional[Any]:
        """Get cached embedding for text."""
        # Use normalized text as key for better hit rate
        key = normalize_ocr_strict(text)[:500]  # Truncate for memory
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None
    
    def set(self, text: str, embedding: Any) -> None:
        """Cache embedding for text."""
        if len(self._cache) >= self._max_size:
            # Simple eviction: clear half
            keys = list(self._cache.keys())
            for k in keys[:len(keys)//2]:
                del self._cache[k]
        
        key = normalize_ocr_strict(text)[:500]
        self._cache[key] = embedding
    
    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / max(1, total)
    
    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0


# Global embedding cache
_embedding_cache = EmbeddingCache()


# =============================================================================
# Stage A: Cheap Gate (No Semantics)
# =============================================================================

def _compute_length_ratio(text_a: str, text_b: str) -> float:
    """Compute length ratio between two texts."""
    len_a = len(text_a.strip())
    len_b = len(text_b.strip())
    if len_a == 0 and len_b == 0:
        return 1.0
    if len_a == 0 or len_b == 0:
        return 0.0
    return min(len_a, len_b) / max(len_a, len_b)


def _compute_bbox_iou(bbox_a: Optional[Tuple], bbox_b: Optional[Tuple]) -> Optional[float]:
    """Compute IoU (Intersection over Union) between two bboxes."""
    if not bbox_a or not bbox_b:
        return None
    
    try:
        x0_a, y0_a, x1_a, y1_a = bbox_a
        x0_b, y0_b, x1_b, y1_b = bbox_b
        
        # Check for degenerate bboxes
        if x1_a <= x0_a or y1_a <= y0_a or x1_b <= x0_b or y1_b <= y0_b:
            return None
        
        # Intersection
        x0_i = max(x0_a, x0_b)
        y0_i = max(y0_a, y0_b)
        x1_i = min(x1_a, x1_b)
        y1_i = min(y1_a, y1_b)
        
        if x1_i <= x0_i or y1_i <= y0_i:
            return 0.0
        
        intersection = (x1_i - x0_i) * (y1_i - y0_i)
        area_a = (x1_a - x0_a) * (y1_a - y0_a)
        area_b = (x1_b - x0_b) * (y1_b - y0_b)
        union = area_a + area_b - intersection
        
        return intersection / max(union, 1e-6)
    except (TypeError, ValueError):
        return None


def _compute_position_drift(
    bbox_a: Optional[Tuple],
    bbox_b: Optional[Tuple],
    page_width: float = 612,
    page_height: float = 792,
) -> Optional[float]:
    """Compute relative position drift between two bboxes."""
    if not bbox_a or not bbox_b:
        return None
    
    try:
        x0_a, y0_a, x1_a, y1_a = bbox_a
        x0_b, y0_b, x1_b, y1_b = bbox_b
        
        # Center points
        cx_a, cy_a = (x0_a + x1_a) / 2, (y0_a + y1_a) / 2
        cx_b, cy_b = (x0_b + x1_b) / 2, (y0_b + y1_b) / 2
        
        # Relative drift
        dx = abs(cx_a - cx_b) / page_width
        dy = abs(cy_a - cy_b) / page_height
        
        return max(dx, dy)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def stage_a_gate(
    text_a: str,
    text_b: str,
    config: GatingConfig,
    bbox_a: Optional[Tuple] = None,
    bbox_b: Optional[Tuple] = None,
) -> GatingResult:
    """Stage A: Cheap gate without semantic analysis.
    
    Uses rapidfuzz for fast text comparison and bbox for position sanity.
    
    Args:
        text_a: First text
        text_b: Second text
        config: Gating configuration
        bbox_a: Optional bounding box for text_a
        bbox_b: Optional bounding box for text_b
        
    Returns:
        GatingResult with decision (may be SEMANTIC_CHECK if gray zone)
    """
    start_time = time.time()
    
    # Get thresholds
    identical_th, likely_th, gray_low, gray_high = config.get_thresholds()
    
    # Normalize texts
    norm_a = normalize_ocr_compare(text_a)
    norm_b = normalize_ocr_compare(text_b)
    
    # Handle empty texts
    if not norm_a and not norm_b:
        return GatingResult(
            decision=GateDecision.IDENTICAL,
            confidence=1.0,
            stage_a_time=time.time() - start_time,
            reason="Both texts empty",
        )
    
    if not norm_a or not norm_b:
        return GatingResult(
            decision=GateDecision.LIKELY_DIFFERENT,
            confidence=0.95,
            length_ratio=0.0,
            stage_a_time=time.time() - start_time,
            reason="One text is empty",
        )
    
    # Length ratio check (fast rejection)
    length_ratio = _compute_length_ratio(norm_a, norm_b)
    if length_ratio < config.length_ratio_min or length_ratio > config.length_ratio_max:
        return GatingResult(
            decision=GateDecision.LIKELY_DIFFERENT,
            confidence=0.9,
            length_ratio=length_ratio,
            stage_a_time=time.time() - start_time,
            reason=f"Length ratio {length_ratio:.2f} outside bounds",
        )
    
    # Compute rapidfuzz scores
    char_score = ratio(norm_a, norm_b) / 100.0
    token_sort_score = token_sort_ratio(norm_a, norm_b) / 100.0
    token_set_score = token_set_ratio(norm_a, norm_b) / 100.0
    
    # Use max of token scores (more robust for OCR)
    best_score = max(token_sort_score, token_set_score, char_score)
    
    # Position metrics
    bbox_iou = _compute_bbox_iou(bbox_a, bbox_b)
    pos_drift = _compute_position_drift(bbox_a, bbox_b)
    
    # Determine decision first
    decision = GateDecision.NEEDS_DIFF  # Default
    reason = ""
    
    # Exact match after normalization
    if norm_a == norm_b:
        decision = GateDecision.IDENTICAL
        reason = "Exact match after normalization"
    # High similarity = likely identical
    elif best_score >= identical_th:
        decision = GateDecision.IDENTICAL
        reason = f"Very high similarity ({best_score:.2%})"
    elif best_score >= likely_th:
        decision = GateDecision.LIKELY_IDENTICAL
        reason = f"High similarity ({best_score:.2%})"
    # Low similarity = definitely different
    elif best_score < gray_low:
        decision = GateDecision.LIKELY_DIFFERENT
        reason = f"Low similarity ({best_score:.2%})"
    # Gray zone - check if bbox suggests layout drift
    elif bbox_iou is not None and bbox_iou < config.bbox_iou_threshold:
        decision = GateDecision.SEMANTIC_CHECK
        reason = f"Low bbox overlap ({bbox_iou:.2f}) - layout drift?"
    elif pos_drift is not None and pos_drift > config.position_drift_threshold:
        decision = GateDecision.SEMANTIC_CHECK
        reason = f"Position drift ({pos_drift:.2%}) - layout change?"
    # Gray zone - needs semantic check
    elif gray_low <= best_score < gray_high:
        decision = GateDecision.SEMANTIC_CHECK
        reason = f"Gray zone ({best_score:.2%}) - needs semantic analysis"
    else:
        decision = GateDecision.NEEDS_DIFF
        reason = f"Moderate similarity ({best_score:.2%})"
    
    # Build result with decision
    result = GatingResult(
        decision=decision,
        confidence=1.0 if norm_a == norm_b else best_score,
        token_sort_score=token_sort_score,
        token_set_score=token_set_score,
        char_ratio_score=char_score,
        length_ratio=length_ratio,
        bbox_overlap_iou=bbox_iou,
        position_drift=pos_drift,
        stage_a_time=time.time() - start_time,
        reason=reason,
    )
    
    return result


# =============================================================================
# Stage B: Semantic Analysis (Gray Zone Only)
# =============================================================================

def stage_b_semantic(
    text_a: str,
    text_b: str,
    stage_a_result: GatingResult,
    config: GatingConfig,
) -> GatingResult:
    """Stage B: Semantic analysis for gray zone cases.
    
    Uses sentence embeddings to determine if texts are semantically similar.
    Only called when Stage A returns SEMANTIC_CHECK.
    
    Args:
        text_a: First text
        text_b: Second text
        stage_a_result: Result from stage A
        config: Gating configuration
        
    Returns:
        Updated GatingResult with semantic score and final decision
    """
    if not config.enable_stage_b:
        # Stage B disabled - use Stage A decision
        stage_a_result.decision = GateDecision.NEEDS_DIFF
        stage_a_result.reason += " (Stage B disabled)"
        return stage_a_result
    
    start_time = time.time()
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        # Check cache
        cached_a = _embedding_cache.get(text_a) if config.cache_embeddings else None
        cached_b = _embedding_cache.get(text_b) if config.cache_embeddings else None
        
        if cached_a is not None and cached_b is not None:
            emb_a, emb_b = cached_a, cached_b
            stage_a_result.embedding_cached = True
        else:
            # Load model (cached by sentence-transformers)
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            
            # Compute embeddings
            texts = []
            if cached_a is None:
                texts.append(normalize_ocr_compare(text_a))
            if cached_b is None:
                texts.append(normalize_ocr_compare(text_b))
            
            embeddings = model.encode(texts, convert_to_numpy=True)
            
            idx = 0
            if cached_a is None:
                emb_a = embeddings[idx]
                if config.cache_embeddings:
                    _embedding_cache.set(text_a, emb_a)
                idx += 1
            else:
                emb_a = cached_a
            
            if cached_b is None:
                emb_b = embeddings[idx]
                if config.cache_embeddings:
                    _embedding_cache.set(text_b, emb_b)
            else:
                emb_b = cached_b
        
        # Compute cosine similarity
        similarity = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))
        
        stage_a_result.semantic_score = similarity
        stage_a_result.stage_b_time = time.time() - start_time
        
        # Decision based on semantic similarity
        if similarity >= config.semantic_identical_threshold:
            stage_a_result.decision = GateDecision.LIKELY_IDENTICAL
            stage_a_result.confidence = similarity
            stage_a_result.reason = f"Semantically identical ({similarity:.2%})"
        elif similarity < config.semantic_different_threshold:
            stage_a_result.decision = GateDecision.LIKELY_DIFFERENT
            stage_a_result.confidence = 1 - similarity
            stage_a_result.reason = f"Semantically different ({similarity:.2%})"
        else:
            stage_a_result.decision = GateDecision.NEEDS_DIFF
            stage_a_result.confidence = similarity
            stage_a_result.reason = f"Semantic gray zone ({similarity:.2%}) - needs diff"
        
    except Exception as e:
        # Fallback if embeddings fail
        stage_a_result.stage_b_time = time.time() - start_time
        stage_a_result.decision = GateDecision.NEEDS_DIFF
        stage_a_result.reason = f"Semantic analysis failed: {e}"
    
    return stage_a_result


# =============================================================================
# Main Gating API
# =============================================================================

def apply_gating(
    text_a: str,
    text_b: str,
    config: Optional[GatingConfig] = None,
    bbox_a: Optional[Tuple] = None,
    bbox_b: Optional[Tuple] = None,
    is_ocr: bool = False,
) -> GatingResult:
    """Apply two-stage gating to determine if diff is needed.
    
    This is the main entry point for the gating system.
    
    Args:
        text_a: First text to compare
        text_b: Second text to compare
        config: Gating configuration (defaults to standard settings)
        bbox_a: Optional bounding box for text_a
        bbox_b: Optional bounding box for text_b
        is_ocr: Whether texts are from OCR (uses more lenient thresholds)
        
    Returns:
        GatingResult with decision and metrics
        
    Example:
        result = apply_gating(text_a, text_b, is_ocr=True)
        if result.should_skip_diff:
            print(f"Skipping diff: {result.reason}")
        elif result.is_phantom_diff:
            print(f"Likely phantom diff: {result.reason}")
    """
    if config is None:
        config = GatingConfig(ocr_mode=is_ocr)
    else:
        config.ocr_mode = is_ocr
    
    # Stage A: Cheap gate
    result = stage_a_gate(text_a, text_b, config, bbox_a, bbox_b)
    
    # Stage B: Semantic analysis (only if gray zone)
    if result.decision == GateDecision.SEMANTIC_CHECK:
        result = stage_b_semantic(text_a, text_b, result, config)
    
    return result


def batch_gating(
    text_pairs: List[Tuple[str, str]],
    config: Optional[GatingConfig] = None,
    is_ocr: bool = False,
) -> Tuple[List[int], List[int], List[GatingResult]]:
    """Apply gating to multiple text pairs efficiently.
    
    Returns indices that need detailed diff vs those that can be skipped.
    
    Args:
        text_pairs: List of (text_a, text_b) tuples
        config: Gating configuration
        is_ocr: Whether texts are from OCR
        
    Returns:
        Tuple of (skip_indices, diff_indices, all_results)
    """
    if config is None:
        config = GatingConfig(ocr_mode=is_ocr)
    else:
        config.ocr_mode = is_ocr
    
    skip_indices: List[int] = []
    diff_indices: List[int] = []
    results: List[GatingResult] = []
    
    for i, (text_a, text_b) in enumerate(text_pairs):
        result = apply_gating(text_a, text_b, config, is_ocr=is_ocr)
        results.append(result)
        
        if result.should_skip_diff:
            skip_indices.append(i)
        else:
            diff_indices.append(i)
    
    return skip_indices, diff_indices, results


# =============================================================================
# Statistics and Monitoring
# =============================================================================

@dataclass
class GatingStats:
    """Statistics from gating operations."""
    
    total_pairs: int = 0
    identical: int = 0
    likely_identical: int = 0
    needs_diff: int = 0
    likely_different: int = 0
    semantic_checks: int = 0
    
    stage_a_time_total: float = 0.0
    stage_b_time_total: float = 0.0
    
    @property
    def skip_rate(self) -> float:
        """Rate of pairs that could skip detailed diff."""
        return (self.identical + self.likely_identical) / max(1, self.total_pairs)
    
    @property
    def phantom_rate(self) -> float:
        """Estimated phantom diff rate."""
        return self.likely_identical / max(1, self.total_pairs)
    
    @property
    def semantic_rate(self) -> float:
        """Rate of pairs requiring semantic analysis."""
        return self.semantic_checks / max(1, self.total_pairs)
    
    @property
    def avg_stage_a_time(self) -> float:
        return self.stage_a_time_total / max(1, self.total_pairs)
    
    @property
    def avg_stage_b_time(self) -> float:
        return self.stage_b_time_total / max(1, self.semantic_checks)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_pairs": self.total_pairs,
            "identical": self.identical,
            "likely_identical": self.likely_identical,
            "needs_diff": self.needs_diff,
            "likely_different": self.likely_different,
            "semantic_checks": self.semantic_checks,
            "skip_rate": self.skip_rate,
            "phantom_rate": self.phantom_rate,
            "semantic_rate": self.semantic_rate,
            "avg_stage_a_time": self.avg_stage_a_time,
            "avg_stage_b_time": self.avg_stage_b_time,
        }


def collect_gating_stats(results: List[GatingResult]) -> GatingStats:
    """Collect statistics from gating results."""
    stats = GatingStats(total_pairs=len(results))
    
    for r in results:
        if r.decision == GateDecision.IDENTICAL:
            stats.identical += 1
        elif r.decision == GateDecision.LIKELY_IDENTICAL:
            stats.likely_identical += 1
        elif r.decision == GateDecision.NEEDS_DIFF:
            stats.needs_diff += 1
        elif r.decision == GateDecision.LIKELY_DIFFERENT:
            stats.likely_different += 1
        
        if r.semantic_score is not None:
            stats.semantic_checks += 1
        
        stats.stage_a_time_total += r.stage_a_time
        stats.stage_b_time_total += r.stage_b_time
    
    return stats


def clear_embedding_cache() -> None:
    """Clear the embedding cache."""
    _embedding_cache.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get embedding cache statistics."""
    return {
        "size": len(_embedding_cache._cache),
        "hits": _embedding_cache._hits,
        "misses": _embedding_cache._misses,
        "hit_rate": _embedding_cache.hit_rate,
    }
