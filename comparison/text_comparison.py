"""Semantic text comparison using embeddings.

Phase 2 Optimization (Step 3):
- Batch encoding with unique text deduplication
- Embedding cache for repeated texts
- Deterministic comparison results
"""
from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING, Set, Tuple

from comparison.alignment import align_pages, align_sections, detect_layout_shift
from comparison.hierarchical_alignment import hierarchical_align
from comparison.models import Diff, PageData
from config.settings import settings
from utils.logging import logger
from utils.text_normalization import normalize_text, compute_ocr_change_significance
from utils.text_diff import detect_character_changes
from utils.diff_projection import (
    get_word_diff_detail,
    detect_layout_drift as detect_layout_drift_regions,
    bbox_union_dict,
)


# NOTE: Using difflib.SequenceMatcher for token-list alignment.
# rapidfuzz only supports string comparison, not list-of-tokens matching with opcodes.
from difflib import SequenceMatcher
import re

def _normalize_token(tok: str) -> str:
    return re.sub(r"\s+", " ", tok.strip().lower())

def _union_bboxes(bboxes: List[Dict[str, float]]) -> Dict[str, float]:
    xs = [b["x"] for b in bboxes]
    ys = [b["y"] for b in bboxes]
    x2s = [b["x"] + b["width"] for b in bboxes]
    y2s = [b["y"] + b["height"] for b in bboxes]
    x0, y0 = min(xs), min(ys)
    x1, y1 = max(x2s), max(y2s)
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}

def _compute_word_level_bboxes(block_a, block_b, page_a: PageData, page_b: PageData) -> dict:
    """Compute word-level bbox lists for a modified diff, using block metadata['words'].

    Returns keys:
      - word_ops: list of {tag, a_tokens, b_tokens}
      - word_bboxes_a: list of normalized bboxes in page A coords
      - word_bboxes_b: list of normalized bboxes in page B coords
      - highlight_mode: "word" or "line_fallback"
    """
    # Use the new get_word_diff_detail function
    detail = get_word_diff_detail(block_a, block_b)
    
    if not detail["ops"] and detail["highlight_mode"] == "line_fallback":
        return {}
    
    from utils.coordinates import normalize_bbox as _norm_bbox
    
    # Normalize bboxes from absolute to relative coordinates
    bboxes_a = []
    for bbox in detail["old_bboxes"]:
        if isinstance(bbox, dict) and "x" in bbox:
            # Convert from {x,y,width,height} to (x0,y0,x1,y1) for normalize_bbox
            x0 = bbox["x"]
            y0 = bbox["y"]
            x1 = bbox["x"] + bbox["width"]
            y1 = bbox["y"] + bbox["height"]
            bboxes_a.append(_norm_bbox((x0, y0, x1, y1), page_a.width, page_a.height))
    
    bboxes_b = []
    for bbox in detail["new_bboxes"]:
        if isinstance(bbox, dict) and "x" in bbox:
            x0 = bbox["x"]
            y0 = bbox["y"]
            x1 = bbox["x"] + bbox["width"]
            y1 = bbox["y"] + bbox["height"]
            bboxes_b.append(_norm_bbox((x0, y0, x1, y1), page_b.width, page_b.height))
    
    word_ops = []
    for op in detail["ops"]:
        word_ops.append({
            "tag": op["tag"],
            "a_tokens": op["old_tokens"],
            "b_tokens": op["new_tokens"],
        })
    
    return {
        "word_ops": word_ops,
        "word_bboxes_a": bboxes_a,
        "word_bboxes_b": bboxes_b,
        "highlight_mode": detail["highlight_mode"],
    }


if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import util  # type: ignore


class TextComparator:
    """Compare text content between documents using semantic embeddings.
    
    Phase 2 Optimizations:
    - Embedding cache for repeated texts within a comparison
    - Batch encoding to reduce model invocations
    - Unique text deduplication before encoding
    """
    
    def __init__(self, model_name: str | None = None, threshold: float | None = None):
        self.model_name = model_name or settings.sentence_transformer_model
        self.threshold = threshold or settings.text_similarity_threshold
        self.ocr_threshold = settings.ocr_text_similarity_threshold
        logger.info("Loading sentence transformer: %s", self.model_name)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name)
        
        # Phase 2: Embedding cache for current comparison
        self._embedding_cache: Dict[str, "torch.Tensor"] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def clear_cache(self) -> None:
        """Clear embedding cache (call between document comparisons)."""
        self._embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get embedding cache statistics."""
        return {
            "cache_size": len(self._embedding_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_ratio": self._cache_hits / max(1, self._cache_hits + self._cache_misses),
        }
    
    def _get_embedding_cached(self, text: str) -> "torch.Tensor":
        """Get embedding from cache or compute and cache it."""
        if text in self._embedding_cache:
            self._cache_hits += 1
            return self._embedding_cache[text]
        
        self._cache_misses += 1
        embedding = self.model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        self._embedding_cache[text] = embedding
        return embedding
    
    def _batch_encode_unique(self, texts: List[str]) -> Dict[str, "torch.Tensor"]:
        """Batch encode unique texts and update cache.
        
        Phase 2 optimization: Deduplicate texts before encoding to minimize
        model invocations.
        """
        # Find unique texts not in cache
        unique_texts = list(set(texts))
        texts_to_encode = [t for t in unique_texts if t and t not in self._embedding_cache]
        
        if texts_to_encode:
            # Batch encode all new texts at once
            embeddings = self.model.encode(
                texts_to_encode, 
                convert_to_tensor=True, 
                show_progress_bar=False,
                batch_size=32,
            )
            
            # Update cache
            for text, embedding in zip(texts_to_encode, embeddings):
                self._embedding_cache[text] = embedding
        
        # Return embeddings for requested texts
        return {t: self._embedding_cache[t] for t in unique_texts if t in self._embedding_cache}
    
    def get_threshold(self, is_ocr: bool = False) -> float:
        """Get the appropriate threshold based on OCR context."""
        return self.ocr_threshold if is_ocr else self.threshold

    def compare(
        self,
        pages_a: List[PageData],
        pages_b: List[PageData],
        alignment_map: dict | None = None,
    ) -> List[Diff]:
        """
        Compare text between two documents and detect differences.
        
        Phase 2: Pre-computes embeddings in batch for all texts before comparing.
        
        Args:
            pages_a: Pages from first document
            pages_b: Pages from second document
            alignment_map: Optional pre-computed page alignment
        
        Returns:
            List of Diff objects representing detected changes
        """
        logger.info("Comparing %d pages vs %d pages", len(pages_a), len(pages_b))
        
        # Clear cache for new comparison
        self.clear_cache()
        
        if alignment_map is None:
            alignment_map = align_pages(pages_a, pages_b, use_similarity=True)
        
        # Phase 2: Collect all texts upfront for batch encoding
        all_texts: Set[str] = set()
        page_b_lookup = {page.page_num: page for page in pages_b}
        
        for page_a in pages_a:
            if page_a.page_num not in alignment_map:
                continue
            
            page_b_num, _ = alignment_map[page_a.page_num]
            if page_b_num not in page_b_lookup:
                continue
            
            page_b = page_b_lookup[page_b_num]
            
            # Get OCR status for normalization
            extraction_method_a = (page_a.metadata or {}).get("extraction_method", "")
            extraction_method_b = (page_b.metadata or {}).get("extraction_method", "")
            is_ocr = ("ocr" in (extraction_method_a or "").lower()) or ("ocr" in (extraction_method_b or "").lower())
            
            # Collect normalized texts from both pages
            for block in page_a.blocks:
                if block.text and block.text.strip():
                    normalized = normalize_text(block.text.strip(), ocr=is_ocr)
                    if normalized:
                        all_texts.add(normalized)
            
            for block in page_b.blocks:
                if block.text and block.text.strip():
                    normalized = normalize_text(block.text.strip(), ocr=is_ocr)
                    if normalized:
                        all_texts.add(normalized)
        
        # Batch encode all unique texts upfront
        if all_texts:
            logger.debug("Phase 2: Batch encoding %d unique texts", len(all_texts))
            self._batch_encode_unique(list(all_texts))
        
        if alignment_map is None:
            alignment_map = align_pages(pages_a, pages_b, use_similarity=True)
        
        all_diffs: List[Diff] = []
        
        # Create page lookup
        page_b_lookup = {page.page_num: page for page in pages_b}
        
        for page_a in pages_a:
            if page_a.page_num not in alignment_map:
                continue
            
            page_b_num, confidence = alignment_map[page_a.page_num]
            if page_b_num not in page_b_lookup:
                # Page doesn't exist in doc_b - mark all as deleted
                for block in page_a.blocks:
                    # Normalize bbox coordinates
                    normalized_bbox = block.normalize_bbox(page_a.width, page_a.height)
                    all_diffs.append(Diff(
                        page_num=page_a.page_num,
                        diff_type="deleted",
                        change_type="content",
                        old_text=block.text,
                        new_text=None,
                        bbox=normalized_bbox,
                        confidence=confidence,
                        metadata={"page_width": page_a.width, "page_height": page_a.height},
                    ))
                continue
            
            page_b = page_b_lookup[page_b_num]

            extraction_method_a = (page_a.metadata or {}).get("extraction_method", "")
            extraction_method_b = (page_b.metadata or {}).get("extraction_method", "")
            is_ocr_page = ("ocr" in (extraction_method_a or "").lower()) or ("ocr" in (extraction_method_b or "").lower())
            
            # Check if hierarchical alignment should be used
            has_markdown_a = page_a.metadata.get("has_markdown_structure", False)
            has_markdown_b = page_b.metadata.get("has_markdown_structure", False)
            use_hierarchical = has_markdown_a or has_markdown_b
            
            # Align blocks within pages
            if use_hierarchical:
                try:
                    # Use hierarchical alignment
                    segment_alignment = hierarchical_align(page_a, page_b, use_dtw=True)
                    # Convert to block alignment for compatibility
                    block_alignment = self._convert_segment_to_block_alignment(
                        page_a, page_b, segment_alignment
                    )
                except Exception as exc:
                    logger.warning("Hierarchical alignment failed, using standard: %s", exc)
                    block_alignment = align_sections(page_a, page_b, use_hierarchical=False)
            else:
                block_alignment = align_sections(page_a, page_b, use_hierarchical=False)
            
            # Compare aligned blocks
            page_diffs = self._compare_page_blocks(
                page_a, page_b, block_alignment, confidence
            )
            all_diffs.extend(page_diffs)

            # Blocks in page_a without matches are deletions.
            matched_a_indices = set(block_alignment.keys())
            for idx_a, block_a in enumerate(page_a.blocks):
                if idx_a in matched_a_indices:
                    continue
                text_a = (block_a.text or "").strip()
                if not text_a:
                    continue
                normalized_bbox = block_a.normalize_bbox(page_a.width, page_a.height)
                all_diffs.append(Diff(
                    page_num=page_a.page_num,
                    diff_type="deleted",
                    change_type="content",
                    old_text=text_a,
                    new_text=None,
                    bbox=normalized_bbox,
                    page_num_b=page_b.page_num,
                    bbox_b=None,
                    confidence=confidence,
                    metadata={"page_width": page_a.width, "page_height": page_a.height, "is_ocr": is_ocr_page},
                ))
            
            # Check for blocks in page_b that don't have matches (additions)
            matched_b_indices = set(block_alignment.values())
            for idx_b, block_b in enumerate(page_b.blocks):
                if idx_b not in matched_b_indices:
                    text_b = (block_b.text or "").strip()
                    if not text_b:
                        continue

                    # Normalize bbox coordinates (B-side)
                    normalized_bbox_b = block_b.normalize_bbox(page_b.width, page_b.height)
                    all_diffs.append(Diff(
                        page_num=page_a.page_num,
                        page_num_b=page_b.page_num,
                        diff_type="added",
                        change_type="content",
                        old_text=None,
                        new_text=text_b,
                        bbox=None,
                        bbox_b=normalized_bbox_b,
                        confidence=confidence,
                        metadata={
                            "page_width": page_a.width,
                            "page_height": page_a.height,
                            "page_width_b": page_b.width,
                            "page_height_b": page_b.height,
                            "is_ocr": is_ocr_page,
                        },
                    ))
        
        # Phase 2: Log embedding cache stats
        cache_stats = self.get_cache_stats()
        logger.debug(
            "Embedding cache: %d entries, %.1f%% hit ratio (%d hits, %d misses)",
            cache_stats["cache_size"],
            cache_stats["hit_ratio"] * 100,
            cache_stats["cache_hits"],
            cache_stats["cache_misses"],
        )
        
        logger.info("Detected %d text differences", len(all_diffs))
        return all_diffs

    def _compare_page_blocks(
        self,
        page_a: PageData,
        page_b: PageData,
        block_alignment: dict,
        confidence: float,
    ) -> List[Diff]:
        """Compare text blocks between two aligned pages."""
        diffs: List[Diff] = []
        
        for idx_a, idx_b in block_alignment.items():
            if idx_a >= len(page_a.blocks) or idx_b >= len(page_b.blocks):
                continue
            
            block_a = page_a.blocks[idx_a]
            block_b = page_b.blocks[idx_b]

            extraction_method_a = (page_a.metadata or {}).get("extraction_method", "")
            extraction_method_b = (page_b.metadata or {}).get("extraction_method", "")
            is_ocr_page = ("ocr" in (extraction_method_a or "").lower()) or ("ocr" in (extraction_method_b or "").lower())
            translation = (page_a.metadata or {}).get("page_alignment_translation") if is_ocr_page else None
            
            # Use OCR-specific threshold
            effective_threshold = self.get_threshold(is_ocr=is_ocr_page)
            
            text_a = block_a.text.strip()
            text_b = block_b.text.strip()
            
            if not text_a and not text_b:
                continue
            
            if not text_a:
                # Text exists only in B -> added
                normalized_bbox_b = block_b.normalize_bbox(page_b.width, page_b.height)
                diffs.append(Diff(
                    page_num=page_a.page_num,
                    page_num_b=page_b.page_num,
                    diff_type="added",
                    change_type="content",
                    old_text=None,
                    new_text=text_b,
                    bbox=None,
                    bbox_b=normalized_bbox_b,
                    confidence=confidence,
                    metadata={
                        "page_width": page_a.width,
                        "page_height": page_a.height,
                        "page_width_b": page_b.width,
                        "page_height_b": page_b.height,
                        "is_ocr": is_ocr_page,
                    },
                ))
                continue
            
            if not text_b:
                # Text exists only in A -> deleted
                normalized_bbox_a = block_a.normalize_bbox(page_a.width, page_a.height)
                diffs.append(Diff(
                    page_num=page_a.page_num,
                    page_num_b=page_b.page_num,
                    diff_type="deleted",
                    change_type="content",
                    old_text=text_a,
                    new_text=None,
                    bbox=normalized_bbox_a,
                    bbox_b=None,
                    confidence=confidence,
                    metadata={"page_width": page_a.width, "page_height": page_a.height, "is_ocr": is_ocr_page},
                ))
                continue
            
            # Normalize text before semantic similarity comparison
            # This ensures case differences and minor variations don't affect comparison
            normalized_a = normalize_text(text_a, ocr=is_ocr_page)
            normalized_b = normalize_text(text_b, ocr=is_ocr_page)
            
            # Compare text similarity using normalized versions
            similarity = self.similarity(normalized_a, normalized_b)
            
            # Check if texts are identical (after normalization)
            texts_identical = normalized_a == normalized_b
            
            # Determine if we need to create a diff
            should_create_diff = False
            character_change_info = None
            ocr_significance = None
            
            if similarity < effective_threshold:
                # Semantic similarity is low - definitely a change
                should_create_diff = True
            elif not texts_identical:
                # Semantic similarity is high but texts differ - check for character-level changes
                character_change_info = detect_character_changes(text_a, text_b)
                if character_change_info["has_character_change"]:
                    # For OCR pages, check if change is significant
                    if is_ocr_page:
                        ocr_significance = compute_ocr_change_significance(text_a, text_b, ocr=True)
                        if ocr_significance["is_significant"]:
                            should_create_diff = True
                        # Else: change is too small, treat as OCR noise
                    else:
                        # Digital PDF: any character change is significant
                        should_create_diff = True
            
            if should_create_diff:
                # Text changed - normalize bbox coordinates
                normalized_bbox = block_a.normalize_bbox(page_a.width, page_a.height)
                
                # Check for layout shift
                layout_shift = detect_layout_shift(
                    block_a,
                    block_b,
                    page_a.width,
                    page_a.height,
                    tolerance_ratio=settings.ocr_layout_tolerance_ratio if is_ocr_page else 0.01,
                    translation=translation,
                )
                
                # Determine change type
                change_type = "content"
                if layout_shift and layout_shift.get("shift_detected"):
                    change_type = "layout"
                
                diff_metadata = {
                    "similarity": float(similarity),
                    "page_width": page_a.width,
                    "page_height": page_a.height,
                    "is_ocr": is_ocr_page,
                    "effective_threshold": effective_threshold,
                }
                # Store both-side bboxes for consistent rendering
                diff_metadata["bbox_a"] = block_a.normalize_bbox(page_a.width, page_a.height)
                diff_metadata["bbox_b"] = block_b.normalize_bbox(page_b.width, page_b.height)

                if layout_shift:
                    diff_metadata["layout_shift"] = layout_shift
                
                # Add character change information if available
                if character_change_info:
                    diff_metadata["character_diff_ratio"] = character_change_info["character_diff_ratio"]
                    diff_metadata["subtype"] = "character_change"
                    # Use character diff ratio for confidence if similarity is high
                    if similarity >= effective_threshold:
                        # High semantic similarity but character change - use character diff for confidence
                        diff_metadata["confidence_source"] = "character_diff"
                
                # Add OCR significance info
                if ocr_significance:
                    diff_metadata["ocr_change_ratio"] = ocr_significance["change_ratio"]
                    diff_metadata["ocr_changed_chars"] = ocr_significance["changed_chars"]
                
                word_meta = _compute_word_level_bboxes(block_a, block_b, page_a, page_b)
                if word_meta:
                    diff_metadata.update(word_meta)
                    # Convenience alias: use page A word bboxes by default
                    diff_metadata["word_bboxes"] = word_meta.get("word_bboxes_a", [])

                diffs.append(Diff(
                    page_num=page_a.page_num,
                    diff_type="modified",
                    change_type=change_type,
                    old_text=text_a,
                    new_text=text_b,
                    bbox=normalized_bbox,
                    page_num_b=page_b.page_num,
                    bbox_b=diff_metadata.get("bbox_b"),
                    confidence=1.0 - similarity if similarity < effective_threshold else max(0.1, character_change_info["character_diff_ratio"] if character_change_info else 0.1),
                    metadata=diff_metadata,
                ))
        
        return diffs

    def similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute semantic similarity between two text strings.
        
        Phase 2: Uses embedding cache to avoid recomputing embeddings.
        
        Returns:
            Cosine similarity score between 0 and 1
        """
        if not text_a or not text_b:
            return 0.0
        
        from sentence_transformers import util

        # Use cached embeddings
        emb_a = self._get_embedding_cached(text_a)
        emb_b = self._get_embedding_cached(text_b)
        return float(util.cos_sim(emb_a, emb_b).item())
    
    def similarity_batch(self, text_pairs: List[tuple[str, str]]) -> List[float]:
        """
        Compute semantic similarity for multiple text pairs in batch.
        
        More efficient than calling similarity() multiple times.
        
        Args:
            text_pairs: List of (text_a, text_b) tuples
        
        Returns:
            List of similarity scores
        """
        if not text_pairs:
            return []
        
        from sentence_transformers import util
        
        texts_a = [pair[0] for pair in text_pairs if pair[0]]
        texts_b = [pair[1] for pair in text_pairs if pair[1]]
        
        if not texts_a or not texts_b or len(texts_a) != len(texts_b):
            # Fallback to individual calls for mismatched pairs
            return [self.similarity(a, b) for a, b in text_pairs]
        
        # Batch encode all texts
        embeddings_a = self.model.encode(texts_a, convert_to_tensor=True, show_progress_bar=False, batch_size=32)
        embeddings_b = self.model.encode(texts_b, convert_to_tensor=True, show_progress_bar=False, batch_size=32)
        
        # Compute cosine similarities in batch
        similarities = util.cos_sim(embeddings_a, embeddings_b)
        
        # Extract diagonal (pairwise similarities)
        return [float(similarities[i][i].item()) for i in range(len(text_pairs))]
    
    def _convert_segment_to_block_alignment(
        self,
        page_a: PageData,
        page_b: PageData,
        segment_alignment: dict,
    ) -> Dict[int, int]:
        """
        Convert segment alignment to block alignment for compatibility.
        
        Args:
            page_a: First page
            page_b: Second page
            segment_alignment: Mapping from segment index in page_a to (segment index in page_b, confidence)
        
        Returns:
            Mapping from block index in page_a to block index in page_b
        """
        from comparison.hierarchical_alignment import segment_document
        
        segments_a = segment_document(page_a)
        segments_b = segment_document(page_b)
        
        block_mapping: Dict[int, int] = {}
        
        for seg_idx_a, (seg_idx_b, _) in segment_alignment.items():
            if seg_idx_a >= len(segments_a) or seg_idx_b >= len(segments_b):
                continue
            
            seg_a = segments_a[seg_idx_a]
            seg_b = segments_b[seg_idx_b]
            
            # Map blocks within segments
            for block_idx_in_seg, block_a in enumerate(seg_a.blocks):
                if block_idx_in_seg < len(seg_b.blocks):
                    try:
                        block_idx_in_page_a = page_a.blocks.index(block_a)
                        block_b = seg_b.blocks[block_idx_in_seg]
                        block_idx_in_page_b = page_b.blocks.index(block_b)
                        block_mapping[block_idx_in_page_a] = block_idx_in_page_b
                    except (ValueError, IndexError):
                        # Block not found, skip
                        continue
        
        return block_mapping