"""Semantic text comparison using embeddings."""
from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

from comparison.alignment import align_pages, align_sections, detect_layout_shift
from comparison.hierarchical_alignment import hierarchical_align
from comparison.models import Diff, PageData
from config.settings import settings
from utils.logging import logger
from utils.text_normalization import normalize_text
from utils.text_diff import detect_character_changes

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import util  # type: ignore


class TextComparator:
    """Compare text content between documents using semantic embeddings."""
    
    def __init__(self, model_name: str | None = None, threshold: float | None = None):
        self.model_name = model_name or settings.sentence_transformer_model
        self.threshold = threshold or settings.text_similarity_threshold
        logger.info("Loading sentence transformer: %s", self.model_name)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name)

    def compare(
        self,
        pages_a: List[PageData],
        pages_b: List[PageData],
        alignment_map: dict | None = None,
    ) -> List[Diff]:
        """
        Compare text between two documents and detect differences.
        
        Args:
            pages_a: Pages from first document
            pages_b: Pages from second document
            alignment_map: Optional pre-computed page alignment
        
        Returns:
            List of Diff objects representing detected changes
        """
        logger.info("Comparing %d pages vs %d pages", len(pages_a), len(pages_b))
        
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
            
            # Check for blocks in page_b that don't have matches (additions)
            matched_b_indices = set(block_alignment.values())
            for idx_b, block_b in enumerate(page_b.blocks):
                if idx_b not in matched_b_indices:
                    # Normalize bbox coordinates
                    normalized_bbox = block_b.normalize_bbox(page_b.width, page_b.height)
                    all_diffs.append(Diff(
                        page_num=page_b.page_num,
                        diff_type="added",
                        change_type="content",
                        old_text=None,
                        new_text=block_b.text,
                        bbox=normalized_bbox,
                        confidence=confidence,
                        metadata={"page_width": page_b.width, "page_height": page_b.height},
                    ))
        
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
            
            text_a = block_a.text.strip()
            text_b = block_b.text.strip()
            
            if not text_a and not text_b:
                continue
            
            if not text_a:
                # Block deleted
                normalized_bbox = block_b.normalize_bbox(page_b.width, page_b.height)
                diffs.append(Diff(
                    page_num=page_a.page_num,
                    diff_type="deleted",
                    change_type="content",
                    old_text=None,
                    new_text=text_b,
                    bbox=normalized_bbox,
                    confidence=confidence,
                    metadata={"page_width": page_b.width, "page_height": page_b.height},
                ))
                continue
            
            if not text_b:
                # Block added
                normalized_bbox = block_a.normalize_bbox(page_a.width, page_a.height)
                diffs.append(Diff(
                    page_num=page_a.page_num,
                    diff_type="added",
                    change_type="content",
                    old_text=text_a,
                    new_text=None,
                    bbox=normalized_bbox,
                    confidence=confidence,
                    metadata={"page_width": page_a.width, "page_height": page_a.height},
                ))
                continue
            
            # Normalize text before semantic similarity comparison
            # This ensures case differences and minor variations don't affect comparison
            normalized_a = normalize_text(text_a)
            normalized_b = normalize_text(text_b)
            
            # Compare text similarity using normalized versions
            similarity = self.similarity(normalized_a, normalized_b)
            
            # Check if texts are identical (after normalization)
            texts_identical = normalized_a == normalized_b
            
            # Determine if we need to create a diff
            should_create_diff = False
            character_change_info = None
            
            if similarity < self.threshold:
                # Semantic similarity is low - definitely a change
                should_create_diff = True
            elif not texts_identical:
                # Semantic similarity is high but texts differ - check for character-level changes
                character_change_info = detect_character_changes(text_a, text_b)
                if character_change_info["has_character_change"]:
                    # Character-level change detected - create diff even if semantic similarity is high
                    should_create_diff = True
            
            if should_create_diff:
                # Text changed - normalize bbox coordinates
                normalized_bbox = block_a.normalize_bbox(page_a.width, page_a.height)
                
                # Check for layout shift
                layout_shift = detect_layout_shift(
                    block_a, block_b, page_a.width, page_a.height
                )
                
                # Determine change type
                change_type = "content"
                if layout_shift and layout_shift.get("shift_detected"):
                    change_type = "layout"
                
                diff_metadata = {
                    "similarity": float(similarity),
                    "page_width": page_a.width,
                    "page_height": page_a.height,
                }
                if layout_shift:
                    diff_metadata["layout_shift"] = layout_shift
                
                # Add character change information if available
                if character_change_info:
                    diff_metadata["character_diff_ratio"] = character_change_info["character_diff_ratio"]
                    diff_metadata["subtype"] = "character_change"
                    # Use character diff ratio for confidence if similarity is high
                    if similarity >= self.threshold:
                        # High semantic similarity but character change - use character diff for confidence
                        diff_metadata["confidence_source"] = "character_diff"
                
                diffs.append(Diff(
                    page_num=page_a.page_num,
                    diff_type="modified",
                    change_type=change_type,
                    old_text=text_a,
                    new_text=text_b,
                    bbox=normalized_bbox,
                    confidence=1.0 - similarity if similarity < self.threshold else max(0.1, character_change_info["character_diff_ratio"] if character_change_info else 0.1),
                    metadata=diff_metadata,
                ))
        
        return diffs

    def similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute semantic similarity between two text strings.
        
        Returns:
            Cosine similarity score between 0 and 1
        """
        if not text_a or not text_b:
            return 0.0
        
        from sentence_transformers import util

        emb_a = self.model.encode(text_a, convert_to_tensor=True, show_progress_bar=False)
        emb_b = self.model.encode(text_b, convert_to_tensor=True, show_progress_bar=False)
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