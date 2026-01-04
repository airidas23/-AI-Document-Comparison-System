"""Unit tests for comparison/text_comparison.py.

Tests cover:
- Helper functions (_normalize_token, _union_bboxes, _compute_word_level_bboxes)
- TextComparator class methods (init, cache, similarity, compare)
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, List

from comparison.models import Diff, PageData, TextBlock, Style


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_bbox() -> Dict[str, float]:
    """Simple bbox for testing."""
    return {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.05}


@pytest.fixture
def sample_style() -> Style:
    """Simple style for testing."""
    return Style(font="Arial", size=12.0, bold=False, italic=False)


@pytest.fixture
def sample_text_block(sample_bbox, sample_style) -> TextBlock:
    """Create a sample text block."""
    return TextBlock(
        text="Hello World",
        bbox=sample_bbox,
        style=sample_style,
        metadata={"words": [{"text": "Hello", "bbox": {"x": 0.1, "y": 0.2, "width": 0.1, "height": 0.05}}]}
    )


@pytest.fixture
def sample_page_data(sample_text_block) -> PageData:
    """Create sample page data."""
    return PageData(
        page_num=1,
        width=612.0,
        height=792.0,
        blocks=[sample_text_block],
        metadata={"extraction_method": "native"}
    )


@pytest.fixture
def empty_page_data() -> PageData:
    """Create empty page data."""
    return PageData(
        page_num=1,
        width=612.0,
        height=792.0,
        blocks=[],
        metadata={}
    )


# =============================================================================
# Tests for _normalize_token
# =============================================================================

class TestNormalizeToken:
    """Tests for _normalize_token function."""
    
    def test_normalize_token_lowercase(self):
        """Test that tokens are lowercased."""
        from comparison.text_comparison import _normalize_token
        assert _normalize_token("HELLO") == "hello"
        assert _normalize_token("HeLLo WoRLD") == "hello world"
    
    def test_normalize_token_strip(self):
        """Test that leading/trailing whitespace is stripped."""
        from comparison.text_comparison import _normalize_token
        assert _normalize_token("  hello  ") == "hello"
        assert _normalize_token("\thello\n") == "hello"
    
    def test_normalize_token_collapse_whitespace(self):
        """Test that multiple spaces are collapsed to single space."""
        from comparison.text_comparison import _normalize_token
        assert _normalize_token("hello   world") == "hello world"
        assert _normalize_token("hello\n\tworld") == "hello world"
    
    def test_normalize_token_empty(self):
        """Test empty string handling."""
        from comparison.text_comparison import _normalize_token
        assert _normalize_token("") == ""
        assert _normalize_token("   ") == ""


# =============================================================================
# Tests for _union_bboxes
# =============================================================================

class TestUnionBboxes:
    """Tests for _union_bboxes function."""
    
    def test_union_single_bbox(self, sample_bbox):
        """Test union of a single bbox returns same bbox."""
        from comparison.text_comparison import _union_bboxes
        result = _union_bboxes([sample_bbox])
        assert result["x"] == pytest.approx(sample_bbox["x"])
        assert result["y"] == pytest.approx(sample_bbox["y"])
        assert result["width"] == pytest.approx(sample_bbox["width"])
        assert result["height"] == pytest.approx(sample_bbox["height"])
    
    def test_union_multiple_bboxes(self):
        """Test union of multiple bboxes."""
        from comparison.text_comparison import _union_bboxes
        bboxes = [
            {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.1},
            {"x": 0.2, "y": 0.2, "width": 0.2, "height": 0.1},
        ]
        result = _union_bboxes(bboxes)
        # Union should be: x=0.1, y=0.1, width=0.3 (0.4-0.1), height=0.2 (0.3-0.1)
        assert result["x"] == pytest.approx(0.1)
        assert result["y"] == pytest.approx(0.1)
        assert result["width"] == pytest.approx(0.3)
        assert result["height"] == pytest.approx(0.2)
    
    def test_union_overlapping_bboxes(self):
        """Test union of overlapping bboxes."""
        from comparison.text_comparison import _union_bboxes
        bboxes = [
            {"x": 0.0, "y": 0.0, "width": 0.5, "height": 0.5},
            {"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5},
        ]
        result = _union_bboxes(bboxes)
        assert result["x"] == pytest.approx(0.0)
        assert result["y"] == pytest.approx(0.0)
        assert result["width"] == pytest.approx(0.75)
        assert result["height"] == pytest.approx(0.75)


# =============================================================================
# Tests for TextComparator
# =============================================================================

class TestTextComparatorInit:
    """Tests for TextComparator initialization."""
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_default_init(self, mock_st):
        """Test default initialization."""
        mock_st.return_value = MagicMock()
        from comparison.text_comparison import TextComparator
        
        comparator = TextComparator()
        assert comparator.model is not None
        assert comparator._embedding_cache == {}
        assert comparator._cache_hits == 0
        assert comparator._cache_misses == 0
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_custom_threshold(self, mock_st):
        """Test initialization with custom threshold."""
        mock_st.return_value = MagicMock()
        from comparison.text_comparison import TextComparator
        
        comparator = TextComparator(threshold=0.9)
        assert comparator.threshold == 0.9
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_custom_model(self, mock_st):
        """Test initialization with custom model name."""
        mock_st.return_value = MagicMock()
        from comparison.text_comparison import TextComparator
        
        comparator = TextComparator(model_name="custom-model")
        assert comparator.model_name == "custom-model"


class TestTextComparatorCache:
    """Tests for TextComparator caching functionality."""
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_clear_cache(self, mock_st):
        """Test cache clearing."""
        mock_st.return_value = MagicMock()
        from comparison.text_comparison import TextComparator
        
        comparator = TextComparator()
        comparator._embedding_cache = {"test": MagicMock()}
        comparator._cache_hits = 5
        comparator._cache_misses = 3
        
        comparator.clear_cache()
        
        assert comparator._embedding_cache == {}
        assert comparator._cache_hits == 0
        assert comparator._cache_misses == 0
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_cache_stats(self, mock_st):
        """Test cache statistics retrieval."""
        mock_st.return_value = MagicMock()
        from comparison.text_comparison import TextComparator
        
        comparator = TextComparator()
        comparator._embedding_cache = {"a": MagicMock(), "b": MagicMock()}
        comparator._cache_hits = 10
        comparator._cache_misses = 5
        
        stats = comparator.get_cache_stats()
        
        assert stats["cache_size"] == 2
        assert stats["cache_hits"] == 10
        assert stats["cache_misses"] == 5
        assert stats["hit_ratio"] == pytest.approx(10/15)
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_cache_stats_empty(self, mock_st):
        """Test cache statistics when empty."""
        mock_st.return_value = MagicMock()
        from comparison.text_comparison import TextComparator
        
        comparator = TextComparator()
        stats = comparator.get_cache_stats()
        
        assert stats["cache_size"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["hit_ratio"] == 0.0


class TestTextComparatorEmbedding:
    """Tests for TextComparator embedding functionality."""
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_get_embedding_cached_miss(self, mock_st):
        """Test embedding cache miss."""
        mock_model = MagicMock()
        mock_embedding = MagicMock()
        mock_model.encode.return_value = mock_embedding
        mock_st.return_value = mock_model
        
        from comparison.text_comparison import TextComparator
        comparator = TextComparator()
        
        result = comparator._get_embedding_cached("test text")
        
        assert result == mock_embedding
        assert comparator._cache_misses == 1
        assert comparator._cache_hits == 0
        assert "test text" in comparator._embedding_cache
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_get_embedding_cached_hit(self, mock_st):
        """Test embedding cache hit."""
        mock_model = MagicMock()
        mock_embedding = MagicMock()
        mock_st.return_value = mock_model
        
        from comparison.text_comparison import TextComparator
        comparator = TextComparator()
        comparator._embedding_cache["test text"] = mock_embedding
        
        result = comparator._get_embedding_cached("test text")
        
        assert result == mock_embedding
        assert comparator._cache_hits == 1
        assert comparator._cache_misses == 0
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_batch_encode_unique(self, mock_st):
        """Test batch encoding of unique texts."""
        import torch
        mock_model = MagicMock()
        mock_embeddings = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        mock_model.encode.return_value = mock_embeddings
        mock_st.return_value = mock_model
        
        from comparison.text_comparison import TextComparator
        comparator = TextComparator()
        
        texts = ["hello", "world", "hello"]  # "hello" is duplicated
        result = comparator._batch_encode_unique(texts)
        
        # Should encode only unique texts
        assert len(result) == 2  # "hello" and "world"
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_batch_encode_uses_cache(self, mock_st):
        """Test that batch encode uses cached embeddings."""
        import torch
        mock_model = MagicMock()
        cached_embedding = torch.tensor([1.0, 2.0])
        mock_st.return_value = mock_model
        
        from comparison.text_comparison import TextComparator
        comparator = TextComparator()
        comparator._embedding_cache["hello"] = cached_embedding
        
        # Only "world" should be encoded
        mock_model.encode.return_value = [torch.tensor([3.0, 4.0])]
        
        texts = ["hello", "world"]
        result = comparator._batch_encode_unique(texts)
        
        assert "hello" in result
        # Check that encode was only called for non-cached texts


class TestTextComparatorSimilarity:
    """Tests for TextComparator similarity computation."""
    
    @patch('sentence_transformers.util.cos_sim')
    @patch('sentence_transformers.SentenceTransformer')
    def test_similarity_identical_texts(self, mock_st, mock_cos_sim):
        """Test similarity of identical texts."""
        import torch
        mock_model = MagicMock()
        mock_embedding = torch.tensor([1.0, 0.0, 0.0])
        mock_model.encode.return_value = mock_embedding
        mock_st.return_value = mock_model
        
        # cos_sim of identical vectors = 1.0
        mock_cos_sim.return_value = torch.tensor([[1.0]])
        
        from comparison.text_comparison import TextComparator
        comparator = TextComparator()
        
        result = comparator.similarity("hello", "hello")
        assert result == pytest.approx(1.0)
    
    @patch('sentence_transformers.util.cos_sim')
    @patch('sentence_transformers.SentenceTransformer')
    def test_similarity_different_texts(self, mock_st, mock_cos_sim):
        """Test similarity of different texts."""
        import torch
        mock_model = MagicMock()
        mock_model.encode.return_value = torch.tensor([1.0, 0.0, 0.0])
        mock_st.return_value = mock_model
        
        mock_cos_sim.return_value = torch.tensor([[0.5]])
        
        from comparison.text_comparison import TextComparator
        comparator = TextComparator()
        
        result = comparator.similarity("hello", "world")
        assert result == pytest.approx(0.5)
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_similarity_empty_text(self, mock_st):
        """Test similarity with empty text."""
        mock_st.return_value = MagicMock()
        
        from comparison.text_comparison import TextComparator
        comparator = TextComparator()
        
        assert comparator.similarity("", "hello") == 0.0
        assert comparator.similarity("hello", "") == 0.0
        assert comparator.similarity("", "") == 0.0


class TestTextComparatorThreshold:
    """Tests for threshold handling."""
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_get_threshold_default(self, mock_st):
        """Test default threshold."""
        mock_st.return_value = MagicMock()
        from comparison.text_comparison import TextComparator
        
        comparator = TextComparator()
        threshold = comparator.get_threshold(is_ocr=False)
        
        assert threshold == comparator.threshold
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_get_threshold_ocr(self, mock_st):
        """Test OCR threshold."""
        mock_st.return_value = MagicMock()
        from comparison.text_comparison import TextComparator
        
        comparator = TextComparator()
        threshold = comparator.get_threshold(is_ocr=True)
        
        assert threshold == comparator.ocr_threshold


class TestTextComparatorCompare:
    """Tests for TextComparator.compare() method."""
    
    @patch('comparison.text_comparison.align_pages')
    @patch('comparison.text_comparison.align_sections')
    @patch('sentence_transformers.SentenceTransformer')
    def test_compare_empty_pages(self, mock_st, mock_align_sections, mock_align_pages, empty_page_data):
        """Test comparison of empty pages."""
        mock_st.return_value = MagicMock()
        mock_align_pages.return_value = {1: (1, 1.0)}
        mock_align_sections.return_value = {}
        
        from comparison.text_comparison import TextComparator
        comparator = TextComparator()
        
        pages_a = [empty_page_data]
        pages_b = [empty_page_data]
        
        diffs = comparator.compare(pages_a, pages_b)
        
        assert isinstance(diffs, list)
    
    @patch('comparison.text_comparison.align_pages')
    @patch('sentence_transformers.SentenceTransformer')
    def test_compare_page_not_in_alignment(self, mock_st, mock_align_pages, sample_page_data):
        """Test handling of pages not in alignment map."""
        mock_st.return_value = MagicMock()
        mock_align_pages.return_value = {}  # No pages aligned
        
        from comparison.text_comparison import TextComparator
        comparator = TextComparator()
        
        pages_a = [sample_page_data]
        pages_b = []
        
        diffs = comparator.compare(pages_a, pages_b)
        
        # Should return empty list since page 1 is not in alignment
        assert isinstance(diffs, list)
    
    @patch('comparison.text_comparison.normalize_text')
    @patch('comparison.text_comparison.align_pages')
    @patch('comparison.text_comparison.align_sections')
    @patch('sentence_transformers.util.cos_sim')
    @patch('sentence_transformers.SentenceTransformer')
    def test_compare_with_alignment_map(self, mock_st, mock_cos_sim, mock_align_sections, 
                                        mock_align_pages, mock_normalize):
        """Test comparison with pre-provided alignment map."""
        import torch
        mock_model = MagicMock()
        mock_model.encode.return_value = torch.tensor([1.0, 0.0])
        mock_st.return_value = mock_model
        mock_cos_sim.return_value = torch.tensor([[1.0]])
        mock_normalize.side_effect = lambda x, **kwargs: x.lower()
        mock_align_sections.return_value = {0: 0}
        
        from comparison.text_comparison import TextComparator
        from comparison.models import PageData, TextBlock
        
        block_a = TextBlock(
            text="Hello World",
            bbox={"x": 10, "y": 20, "width": 100, "height": 20},
            metadata={}
        )
        block_b = TextBlock(
            text="Hello World",
            bbox={"x": 10, "y": 20, "width": 100, "height": 20},
            metadata={}
        )
        
        page_a = PageData(page_num=1, width=612, height=792, blocks=[block_a], metadata={})
        page_b = PageData(page_num=1, width=612, height=792, blocks=[block_b], metadata={})
        
        comparator = TextComparator()
        
        alignment_map = {1: (1, 1.0)}
        diffs = comparator.compare([page_a], [page_b], alignment_map=alignment_map)
        
        # Should not be called if alignment_map is provided
        mock_align_pages.assert_not_called()


class TestTextComparatorComparePageBlocks:
    """Tests for TextComparator._compare_page_blocks() method."""
    
    @patch('comparison.text_comparison.normalize_text')
    @patch('sentence_transformers.util.cos_sim')
    @patch('sentence_transformers.SentenceTransformer')
    def test_compare_identical_blocks(self, mock_st, mock_cos_sim, mock_normalize):
        """Test comparing identical blocks returns no diffs."""
        import torch
        mock_model = MagicMock()
        mock_model.encode.return_value = torch.tensor([1.0, 0.0])
        mock_st.return_value = mock_model
        mock_cos_sim.return_value = torch.tensor([[1.0]])  # Perfect similarity
        mock_normalize.side_effect = lambda x, **kwargs: x.lower()
        
        from comparison.text_comparison import TextComparator
        from comparison.models import PageData, TextBlock
        
        block = TextBlock(
            text="Hello World",
            bbox={"x": 10, "y": 20, "width": 100, "height": 20},
            metadata={}
        )
        
        page_a = PageData(page_num=1, width=612, height=792, blocks=[block], metadata={})
        page_b = PageData(page_num=1, width=612, height=792, blocks=[block], metadata={})
        
        comparator = TextComparator()
        
        diffs = comparator._compare_page_blocks(page_a, page_b, {0: 0}, confidence=1.0)
        
        # Identical blocks should produce no diffs
        assert len(diffs) == 0
    
    @patch('comparison.text_comparison.normalize_text')
    @patch('sentence_transformers.util.cos_sim')
    @patch('sentence_transformers.SentenceTransformer')
    def test_compare_modified_blocks(self, mock_st, mock_cos_sim, mock_normalize):
        """Test comparing modified blocks returns modified diff."""
        import torch
        mock_model = MagicMock()
        mock_model.encode.return_value = torch.tensor([1.0, 0.0])
        mock_st.return_value = mock_model
        mock_cos_sim.return_value = torch.tensor([[0.5]])  # Low similarity
        mock_normalize.side_effect = lambda x, **kwargs: x.lower()
        
        from comparison.text_comparison import TextComparator
        from comparison.models import PageData, TextBlock
        
        block_a = TextBlock(
            text="Hello World",
            bbox={"x": 10, "y": 20, "width": 100, "height": 20},
            metadata={}
        )
        block_b = TextBlock(
            text="Goodbye World",
            bbox={"x": 10, "y": 20, "width": 100, "height": 20},
            metadata={}
        )
        
        page_a = PageData(page_num=1, width=612, height=792, blocks=[block_a], metadata={})
        page_b = PageData(page_num=1, width=612, height=792, blocks=[block_b], metadata={})
        
        comparator = TextComparator()
        
        diffs = comparator._compare_page_blocks(page_a, page_b, {0: 0}, confidence=1.0)
        
        assert len(diffs) == 1
        assert diffs[0].diff_type == "modified"
        assert diffs[0].old_text == "Hello World"
        assert diffs[0].new_text == "Goodbye World"
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_compare_empty_block_a(self, mock_st):
        """Test comparing when block A has empty text."""
        mock_st.return_value = MagicMock()
        
        from comparison.text_comparison import TextComparator
        from comparison.models import PageData, TextBlock
        
        block_a = TextBlock(
            text="",
            bbox={"x": 10, "y": 20, "width": 100, "height": 20},
            metadata={}
        )
        block_b = TextBlock(
            text="New Text",
            bbox={"x": 10, "y": 20, "width": 100, "height": 20},
            metadata={}
        )
        
        page_a = PageData(page_num=1, width=612, height=792, blocks=[block_a], metadata={})
        page_b = PageData(page_num=1, width=612, height=792, blocks=[block_b], metadata={})
        
        comparator = TextComparator()
        
        diffs = comparator._compare_page_blocks(page_a, page_b, {0: 0}, confidence=1.0)
        
        assert len(diffs) == 1
        assert diffs[0].diff_type == "added"
        assert diffs[0].new_text == "New Text"
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_compare_empty_block_b(self, mock_st):
        """Test comparing when block B has empty text."""
        mock_st.return_value = MagicMock()
        
        from comparison.text_comparison import TextComparator
        from comparison.models import PageData, TextBlock
        
        block_a = TextBlock(
            text="Old Text",
            bbox={"x": 10, "y": 20, "width": 100, "height": 20},
            metadata={}
        )
        block_b = TextBlock(
            text="",
            bbox={"x": 10, "y": 20, "width": 100, "height": 20},
            metadata={}
        )
        
        page_a = PageData(page_num=1, width=612, height=792, blocks=[block_a], metadata={})
        page_b = PageData(page_num=1, width=612, height=792, blocks=[block_b], metadata={})
        
        comparator = TextComparator()
        
        diffs = comparator._compare_page_blocks(page_a, page_b, {0: 0}, confidence=1.0)
        
        assert len(diffs) == 1
        assert diffs[0].diff_type == "deleted"
        assert diffs[0].old_text == "Old Text"


class TestTextComparatorSimilarityBatch:
    """Tests for TextComparator.similarity_batch() method."""
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_similarity_batch_empty(self, mock_st):
        """Test batch similarity with empty list."""
        mock_st.return_value = MagicMock()
        
        from comparison.text_comparison import TextComparator
        comparator = TextComparator()
        
        result = comparator.similarity_batch([])
        assert result == []
    
    @patch('sentence_transformers.util.cos_sim')
    @patch('sentence_transformers.SentenceTransformer')
    def test_similarity_batch_single_pair(self, mock_st, mock_cos_sim):
        """Test batch similarity with single pair."""
        import torch
        mock_model = MagicMock()
        mock_model.encode.return_value = torch.tensor([[1.0, 0.0]])
        mock_st.return_value = mock_model
        
        # Create a mock tensor that supports diagonal access
        mock_result = torch.tensor([[0.8]])
        mock_cos_sim.return_value = mock_result
        
        from comparison.text_comparison import TextComparator
        comparator = TextComparator()
        
        result = comparator.similarity_batch([("hello", "world")])
        
        assert len(result) == 1
        assert result[0] == pytest.approx(0.8)


class TestComputeWordLevelBboxes:
    """Tests for _compute_word_level_bboxes function."""
    
    @patch('comparison.text_comparison.get_word_diff_detail')
    def test_compute_word_bboxes_empty_ops(self, mock_get_detail):
        """Test handling when no ops are returned."""
        mock_get_detail.return_value = {
            "ops": [],
            "old_bboxes": [],
            "new_bboxes": [],
            "highlight_mode": "line_fallback"
        }
        
        from comparison.text_comparison import _compute_word_level_bboxes
        from comparison.models import PageData, TextBlock
        
        block_a = TextBlock(text="Hello", bbox={"x": 0, "y": 0, "width": 100, "height": 20}, metadata={})
        block_b = TextBlock(text="World", bbox={"x": 0, "y": 0, "width": 100, "height": 20}, metadata={})
        
        page_a = PageData(page_num=1, width=612, height=792, blocks=[block_a], metadata={})
        page_b = PageData(page_num=1, width=612, height=792, blocks=[block_b], metadata={})
        
        result = _compute_word_level_bboxes(block_a, block_b, page_a, page_b)
        
        assert result == {}
    
    @patch('comparison.text_comparison.get_word_diff_detail')
    def test_compute_word_bboxes_with_ops(self, mock_get_detail):
        """Test word-level bbox computation with valid ops."""
        mock_get_detail.return_value = {
            "ops": [
                {"tag": "replace", "old_tokens": ["Hello"], "new_tokens": ["World"]}
            ],
            "old_bboxes": [{"x": 10, "y": 20, "width": 50, "height": 20}],
            "new_bboxes": [{"x": 10, "y": 20, "width": 60, "height": 20}],
            "highlight_mode": "word"
        }
        
        from comparison.text_comparison import _compute_word_level_bboxes
        from comparison.models import PageData, TextBlock
        
        block_a = TextBlock(text="Hello", bbox={"x": 0, "y": 0, "width": 100, "height": 20}, metadata={})
        block_b = TextBlock(text="World", bbox={"x": 0, "y": 0, "width": 100, "height": 20}, metadata={})
        
        page_a = PageData(page_num=1, width=612, height=792, blocks=[block_a], metadata={})
        page_b = PageData(page_num=1, width=612, height=792, blocks=[block_b], metadata={})
        
        result = _compute_word_level_bboxes(block_a, block_b, page_a, page_b)
        
        assert "word_ops" in result
        assert "word_bboxes_a" in result
        assert "word_bboxes_b" in result
        assert result["highlight_mode"] == "word"
