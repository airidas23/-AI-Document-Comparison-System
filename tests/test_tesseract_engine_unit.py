"""Unit tests for extraction/tesseract_ocr_engine.py.

Tests cover:
- Pixmap to PIL conversion (_pixmap_to_pil)
- Sanitization functions (_sanitize_psm_mode, _sanitize_oem_mode, etc.)
- Config building (_build_tesseract_config)
- Confidence parsing (_parse_confidence)
- Bbox operations (_build_bbox, _merge_bboxes)
- Image preprocessing (_preprocess_for_tesseract)
- Bbox tightening (_tighten_bbox_to_ink)
- Word grouping (_group_word_entries)
- OCR data conversion (_tesseract_data_to_text_blocks)
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, List
import numpy as np
from PIL import Image


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_bbox() -> Dict[str, float]:
    """Simple bbox for testing."""
    return {"x": 10.0, "y": 20.0, "width": 50.0, "height": 30.0}


@pytest.fixture
def sample_grayscale_image():
    """Create a sample grayscale image."""
    return Image.new("L", (100, 100), color=200)


@pytest.fixture
def sample_rgb_image():
    """Create a sample RGB image."""
    return Image.new("RGB", (100, 100), color=(255, 255, 255))


# =============================================================================
# Tests for _coerce_int
# =============================================================================

class TestCoerceInt:
    """Tests for _coerce_int function."""
    
    def test_coerce_int_valid(self):
        """Test coercion of valid integers."""
        from extraction.tesseract_ocr_engine import _coerce_int
        
        assert _coerce_int(5, 0) == 5
        assert _coerce_int("10", 0) == 10
        assert _coerce_int(3.7, 0) == 3
    
    def test_coerce_int_invalid(self):
        """Test coercion of invalid values returns default."""
        from extraction.tesseract_ocr_engine import _coerce_int
        
        assert _coerce_int("abc", 42) == 42
        assert _coerce_int(None, 99) == 99
        assert _coerce_int([], 5) == 5


# =============================================================================
# Tests for sanitization functions
# =============================================================================

class TestSanitizePsmMode:
    """Tests for _sanitize_psm_mode function."""
    
    def test_sanitize_psm_valid(self):
        """Test valid PSM values."""
        from extraction.tesseract_ocr_engine import _sanitize_psm_mode
        
        assert _sanitize_psm_mode(3) == 3
        assert _sanitize_psm_mode(6) == 6
        assert _sanitize_psm_mode(0) == 0
        assert _sanitize_psm_mode(13) == 13
    
    def test_sanitize_psm_invalid(self):
        """Test invalid PSM values return default."""
        from extraction.tesseract_ocr_engine import _sanitize_psm_mode, DEFAULT_PSM_MODE
        
        assert _sanitize_psm_mode(-1) == DEFAULT_PSM_MODE
        assert _sanitize_psm_mode(14) == DEFAULT_PSM_MODE
        assert _sanitize_psm_mode(None) == DEFAULT_PSM_MODE


class TestSanitizeOemMode:
    """Tests for _sanitize_oem_mode function."""
    
    def test_sanitize_oem_valid(self):
        """Test valid OEM values."""
        from extraction.tesseract_ocr_engine import _sanitize_oem_mode
        
        assert _sanitize_oem_mode(0) == 0
        assert _sanitize_oem_mode(1) == 1
        assert _sanitize_oem_mode(3) == 3
    
    def test_sanitize_oem_invalid(self):
        """Test invalid OEM values return default."""
        from extraction.tesseract_ocr_engine import _sanitize_oem_mode, DEFAULT_OEM_MODE
        
        assert _sanitize_oem_mode(-1) == DEFAULT_OEM_MODE
        assert _sanitize_oem_mode(4) == DEFAULT_OEM_MODE
        assert _sanitize_oem_mode(None) == DEFAULT_OEM_MODE


class TestSanitizeMinConfidence:
    """Tests for _sanitize_min_confidence function."""
    
    def test_sanitize_confidence_valid(self):
        """Test valid confidence values."""
        from extraction.tesseract_ocr_engine import _sanitize_min_confidence
        
        assert _sanitize_min_confidence(50) == 50
        assert _sanitize_min_confidence(0) == 0
        assert _sanitize_min_confidence(100) == 100
    
    def test_sanitize_confidence_clamped(self):
        """Test confidence values are clamped to [0, 100]."""
        from extraction.tesseract_ocr_engine import _sanitize_min_confidence
        
        assert _sanitize_min_confidence(-10) == 0
        assert _sanitize_min_confidence(150) == 100


class TestSanitizeGranularity:
    """Tests for _sanitize_granularity function."""
    
    def test_sanitize_granularity_valid(self):
        """Test valid granularity values."""
        from extraction.tesseract_ocr_engine import _sanitize_granularity
        
        assert _sanitize_granularity("word") == "word"
        assert _sanitize_granularity("line") == "line"
        assert _sanitize_granularity("paragraph") == "paragraph"
        assert _sanitize_granularity("block") == "block"
    
    def test_sanitize_granularity_invalid(self):
        """Test invalid granularity values return 'word'."""
        from extraction.tesseract_ocr_engine import _sanitize_granularity
        
        assert _sanitize_granularity("invalid") == "word"
        assert _sanitize_granularity(None) == "word"
        assert _sanitize_granularity("") == "word"


# =============================================================================
# Tests for _build_tesseract_config
# =============================================================================

class TestBuildTesseractConfig:
    """Tests for _build_tesseract_config function."""
    
    def test_build_config_default(self):
        """Test default config string."""
        from extraction.tesseract_ocr_engine import _build_tesseract_config
        
        config = _build_tesseract_config(3, 3, "", dpi=300)
        
        assert "--psm 3" in config
        assert "--oem 3" in config
        assert "--dpi 300" in config
    
    def test_build_config_with_extra(self):
        """Test config with extra options."""
        from extraction.tesseract_ocr_engine import _build_tesseract_config
        
        config = _build_tesseract_config(6, 1, "-c preserve_interword_spaces=1", dpi=150)
        
        assert "--psm 6" in config
        assert "--oem 1" in config
        assert "--dpi 150" in config
        assert "preserve_interword_spaces=1" in config
    
    def test_build_config_removes_duplicates(self):
        """Test that extra config duplicates are removed."""
        from extraction.tesseract_ocr_engine import _build_tesseract_config
        
        # Extra config with duplicate psm should be cleaned
        config = _build_tesseract_config(3, 3, "--psm 6 -c test=1", dpi=300)
        
        # Should only have one --psm 3 (the main one)
        assert config.count("--psm") == 1
        assert "-c test=1" in config


# =============================================================================
# Tests for _parse_confidence
# =============================================================================

class TestParseConfidence:
    """Tests for _parse_confidence function."""
    
    def test_parse_valid_confidence(self):
        """Test parsing valid confidence values."""
        from extraction.tesseract_ocr_engine import _parse_confidence
        
        conf_i, conf_f = _parse_confidence(85)
        assert conf_i == 85
        assert conf_f == pytest.approx(0.85)
        
        conf_i, conf_f = _parse_confidence("90")
        assert conf_i == 90
        assert conf_f == pytest.approx(0.90)
    
    def test_parse_confidence_bounds(self):
        """Test confidence is bounded correctly."""
        from extraction.tesseract_ocr_engine import _parse_confidence
        
        conf_i, conf_f = _parse_confidence(100)
        assert conf_f == pytest.approx(1.0)
        
        conf_i, conf_f = _parse_confidence(0)
        assert conf_f == 0.0
    
    def test_parse_invalid_confidence(self):
        """Test parsing invalid confidence returns 0."""
        from extraction.tesseract_ocr_engine import _parse_confidence
        
        conf_i, conf_f = _parse_confidence("invalid")
        assert conf_i == 0
        assert conf_f == 0.0
        
        conf_i, conf_f = _parse_confidence(None)
        assert conf_i == 0
    
    def test_parse_negative_confidence(self):
        """Test negative confidence is clamped to 0."""
        from extraction.tesseract_ocr_engine import _parse_confidence
        
        conf_i, conf_f = _parse_confidence(-10)
        assert conf_i == 0
        assert conf_f == 0.0


# =============================================================================
# Tests for _build_bbox
# =============================================================================

class TestBuildBbox:
    """Tests for _build_bbox function."""
    
    def test_build_bbox_valid(self):
        """Test building valid bbox."""
        from extraction.tesseract_ocr_engine import _build_bbox
        
        bbox = _build_bbox(10, 20, 50, 30, scale_factor=1.0)
        
        assert bbox == {"x": 10.0, "y": 20.0, "width": 50.0, "height": 30.0}
    
    def test_build_bbox_with_scale(self):
        """Test building bbox with scale factor."""
        from extraction.tesseract_ocr_engine import _build_bbox
        
        bbox = _build_bbox(100, 200, 50, 30, scale_factor=0.5)
        
        assert bbox["x"] == 50.0
        assert bbox["y"] == 100.0
        assert bbox["width"] == 25.0
        assert bbox["height"] == 15.0
    
    def test_build_bbox_invalid_dimensions(self):
        """Test building bbox with invalid dimensions returns None."""
        from extraction.tesseract_ocr_engine import _build_bbox
        
        assert _build_bbox(10, 20, 0, 30, 1.0) is None
        assert _build_bbox(10, 20, 50, 0, 1.0) is None
        assert _build_bbox(10, 20, -5, 30, 1.0) is None
    
    def test_build_bbox_invalid_values(self):
        """Test building bbox with non-numeric values returns None."""
        from extraction.tesseract_ocr_engine import _build_bbox
        
        assert _build_bbox("abc", 20, 50, 30, 1.0) is None
        assert _build_bbox(10, None, 50, 30, 1.0) is None


# =============================================================================
# Tests for _merge_bboxes
# =============================================================================

class TestMergeBboxes:
    """Tests for _merge_bboxes function."""
    
    def test_merge_empty_list(self):
        """Test merging empty list returns None."""
        from extraction.tesseract_ocr_engine import _merge_bboxes
        
        assert _merge_bboxes([]) is None
    
    def test_merge_single_bbox(self, sample_bbox):
        """Test merging single bbox returns same bbox."""
        from extraction.tesseract_ocr_engine import _merge_bboxes
        
        result = _merge_bboxes([sample_bbox])
        
        assert result["x"] == sample_bbox["x"]
        assert result["y"] == sample_bbox["y"]
        assert result["width"] == sample_bbox["width"]
        assert result["height"] == sample_bbox["height"]
    
    def test_merge_multiple_bboxes(self):
        """Test merging multiple bboxes."""
        from extraction.tesseract_ocr_engine import _merge_bboxes
        
        bboxes = [
            {"x": 10, "y": 10, "width": 20, "height": 20},
            {"x": 50, "y": 50, "width": 20, "height": 20},
        ]
        result = _merge_bboxes(bboxes)
        
        # Merged: x=10, y=10, width=60 (70-10), height=60 (70-10)
        assert result["x"] == 10
        assert result["y"] == 10
        assert result["width"] == 60
        assert result["height"] == 60


# =============================================================================
# Tests for _preprocess_for_tesseract
# =============================================================================

class TestPreprocessForTesseract:
    """Tests for _preprocess_for_tesseract function."""
    
    @patch('config.settings.settings')
    def test_preprocess_disabled(self, mock_settings, sample_rgb_image):
        """Test preprocessing when disabled."""
        mock_settings.tesseract_preprocess_enabled = False
        
        from extraction.tesseract_ocr_engine import _preprocess_for_tesseract
        
        result = _preprocess_for_tesseract(sample_rgb_image)
        
        assert result.mode == "L"  # Should be grayscale
    
    @patch('config.settings.settings')
    def test_preprocess_default(self, mock_settings, sample_rgb_image):
        """Test default preprocessing."""
        mock_settings.tesseract_preprocess_enabled = True
        mock_settings.tesseract_preprocess_median_size = 3
        mock_settings.tesseract_preprocess_unsharp = True
        mock_settings.tesseract_preprocess_binarize = False
        mock_settings.tesseract_invert = False
        
        from extraction.tesseract_ocr_engine import _preprocess_for_tesseract
        
        result = _preprocess_for_tesseract(sample_rgb_image)
        
        assert result.mode == "L"
        assert result.size == sample_rgb_image.size
    
    @patch('config.settings.settings')
    def test_preprocess_with_invert(self, mock_settings, sample_rgb_image):
        """Test preprocessing with inversion."""
        mock_settings.tesseract_preprocess_enabled = True
        mock_settings.tesseract_preprocess_median_size = 3
        mock_settings.tesseract_preprocess_unsharp = False
        mock_settings.tesseract_preprocess_binarize = False
        mock_settings.tesseract_invert = True
        
        from extraction.tesseract_ocr_engine import _preprocess_for_tesseract
        
        result = _preprocess_for_tesseract(sample_rgb_image)
        
        # Result should be inverted
        assert result.mode == "L"


# =============================================================================
# Tests for _tighten_bbox_to_ink
# =============================================================================

class TestTightenBboxToInk:
    """Tests for _tighten_bbox_to_ink function."""
    
    def test_tighten_no_ink(self, sample_grayscale_image):
        """Test tightening when no ink (all white)."""
        from extraction.tesseract_ocr_engine import _tighten_bbox_to_ink
        
        bbox = {"x": 10, "y": 10, "width": 50, "height": 30}
        result = _tighten_bbox_to_ink(sample_grayscale_image, bbox, dpi=72)
        
        # No ink found, should return original bbox
        assert result == bbox
    
    def test_tighten_with_ink(self):
        """Test tightening when ink is present."""
        from extraction.tesseract_ocr_engine import _tighten_bbox_to_ink
        
        # Create image with some dark pixels
        img = Image.new("L", (100, 100), color=255)  # White background
        pixels = img.load()
        # Add some ink (dark pixels) in center
        for y in range(40, 60):
            for x in range(40, 60):
                pixels[x, y] = 50  # Dark
        
        bbox = {"x": 10, "y": 10, "width": 80, "height": 80}
        result = _tighten_bbox_to_ink(img, bbox, dpi=72, pad_px=1)
        
        # Result should be tightened to ink region
        assert result is not None


# =============================================================================
# Tests for _group_word_entries
# =============================================================================

class TestGroupWordEntries:
    """Tests for _group_word_entries function."""
    
    def test_group_by_line(self):
        """Test grouping by line."""
        from extraction.tesseract_ocr_engine import _group_word_entries
        
        entries = [
            {"text": "Hello", "bbox": {"x": 10, "y": 10, "width": 30, "height": 15}, "conf": 0.9,
             "block_num": 1, "par_num": 1, "line_num": 1, "word_num": 1, "order": 0},
            {"text": "World", "bbox": {"x": 50, "y": 10, "width": 30, "height": 15}, "conf": 0.85,
             "block_num": 1, "par_num": 1, "line_num": 1, "word_num": 2, "order": 1},
            {"text": "Next", "bbox": {"x": 10, "y": 30, "width": 30, "height": 15}, "conf": 0.95,
             "block_num": 1, "par_num": 1, "line_num": 2, "word_num": 1, "order": 2},
        ]
        
        result = _group_word_entries(entries, "line")
        
        # Should have 2 groups (2 lines)
        assert len(result) == 2
        assert "Hello World" in [g["text"] for g in result]
        assert "Next" in [g["text"] for g in result]
    
    def test_group_by_paragraph(self):
        """Test grouping by paragraph."""
        from extraction.tesseract_ocr_engine import _group_word_entries
        
        entries = [
            {"text": "Word1", "bbox": {"x": 10, "y": 10, "width": 30, "height": 15}, "conf": 0.9,
             "block_num": 1, "par_num": 1, "line_num": 1, "word_num": 1, "order": 0},
            {"text": "Word2", "bbox": {"x": 50, "y": 10, "width": 30, "height": 15}, "conf": 0.85,
             "block_num": 1, "par_num": 1, "line_num": 2, "word_num": 1, "order": 1},
        ]
        
        result = _group_word_entries(entries, "paragraph")
        
        # Should have 1 group (same paragraph)
        assert len(result) == 1
        assert result[0]["text"] == "Word1 Word2"
    
    def test_group_empty_entries(self):
        """Test grouping empty entries."""
        from extraction.tesseract_ocr_engine import _group_word_entries
        
        result = _group_word_entries([], "line")
        
        assert result == []


# =============================================================================
# Tests for _pixmap_to_pil
# =============================================================================

class TestPixmapToPil:
    """Tests for _pixmap_to_pil function."""
    
    def test_grayscale_conversion(self):
        """Test conversion of grayscale pixmap."""
        from extraction.tesseract_ocr_engine import _pixmap_to_pil
        
        mock_pix = MagicMock()
        mock_pix.width = 50
        mock_pix.height = 50
        mock_pix.n = 1
        mock_pix.alpha = 0
        mock_pix.samples = bytes([128] * (50 * 50))  # Grayscale
        
        result = _pixmap_to_pil(mock_pix)
        
        assert result.mode == "L"
        assert result.size == (50, 50)
    
    def test_rgb_conversion(self):
        """Test conversion of RGB pixmap."""
        from extraction.tesseract_ocr_engine import _pixmap_to_pil
        
        mock_pix = MagicMock()
        mock_pix.width = 50
        mock_pix.height = 50
        mock_pix.n = 3
        mock_pix.alpha = 0
        mock_pix.samples = bytes([128] * (50 * 50 * 3))  # RGB
        
        result = _pixmap_to_pil(mock_pix)
        
        assert result.mode == "RGB"
        assert result.size == (50, 50)
    
    def test_rgba_conversion(self):
        """Test conversion of RGBA pixmap."""
        from extraction.tesseract_ocr_engine import _pixmap_to_pil
        
        mock_pix = MagicMock()
        mock_pix.width = 50
        mock_pix.height = 50
        mock_pix.n = 4
        mock_pix.alpha = 1
        mock_pix.samples = bytes([128] * (50 * 50 * 4))  # RGBA
        
        result = _pixmap_to_pil(mock_pix)
        
        # Should convert RGBA to RGB
        assert result.mode == "RGB"
        assert result.size == (50, 50)


# =============================================================================
# Tests for _tesseract_data_to_text_blocks
# =============================================================================

class TestTesseractDataToTextBlocks:
    """Tests for _tesseract_data_to_text_blocks function."""
    
    def test_convert_empty_data(self):
        """Test conversion of empty OCR data."""
        from extraction.tesseract_ocr_engine import _tesseract_data_to_text_blocks
        
        result = _tesseract_data_to_text_blocks({}, dpi=300, granularity="word", min_confidence=30)
        
        assert result == []
    
    def test_convert_no_text_key(self):
        """Test conversion of data without 'text' key."""
        from extraction.tesseract_ocr_engine import _tesseract_data_to_text_blocks
        
        result = _tesseract_data_to_text_blocks({"level": [5]}, dpi=300, granularity="word", min_confidence=30)
        
        assert result == []
    
    def test_convert_word_granularity(self):
        """Test conversion with word granularity."""
        from extraction.tesseract_ocr_engine import _tesseract_data_to_text_blocks
        
        ocr_data = {
            "text": ["Hello", "World"],
            "left": [10, 50],
            "top": [10, 10],
            "width": [30, 35],
            "height": [15, 15],
            "conf": [90, 85],
            "level": [5, 5],
            "block_num": [1, 1],
            "par_num": [1, 1],
            "line_num": [1, 1],
            "word_num": [1, 2],
        }
        
        result = _tesseract_data_to_text_blocks(ocr_data, dpi=72, granularity="word", min_confidence=30)
        
        assert len(result) == 2
        assert result[0].text == "Hello"
        assert result[1].text == "World"
    
    def test_convert_line_granularity(self):
        """Test conversion with line granularity."""
        from extraction.tesseract_ocr_engine import _tesseract_data_to_text_blocks
        
        ocr_data = {
            "text": ["Hello", "World"],
            "left": [10, 50],
            "top": [10, 10],
            "width": [30, 35],
            "height": [15, 15],
            "conf": [90, 85],
            "level": [5, 5],
            "block_num": [1, 1],
            "par_num": [1, 1],
            "line_num": [1, 1],
            "word_num": [1, 2],
        }
        
        result = _tesseract_data_to_text_blocks(ocr_data, dpi=72, granularity="line", min_confidence=30)
        
        # Should group into single line
        assert len(result) == 1
        assert "Hello World" == result[0].text
    
    def test_convert_empty_words_filtered(self):
        """Test that empty words are filtered."""
        from extraction.tesseract_ocr_engine import _tesseract_data_to_text_blocks
        
        ocr_data = {
            "text": ["Hello", "", "World"],
            "left": [10, 30, 50],
            "top": [10, 10, 10],
            "width": [20, 10, 35],
            "height": [15, 15, 15],
            "conf": [90, 0, 85],
            "level": [5, 5, 5],
            "block_num": [1, 1, 1],
            "par_num": [1, 1, 1],
            "line_num": [1, 1, 1],
            "word_num": [1, 2, 3],
        }
        
        result = _tesseract_data_to_text_blocks(ocr_data, dpi=72, granularity="word", min_confidence=30)
        
        # Empty word should be filtered
        assert len(result) == 2


# =============================================================================
# Tests for ocr_pdf (integration-style with mocking)
# =============================================================================

class TestOcrPdf:
    """Tests for ocr_pdf function (with heavy mocking)."""
    
    @patch('fitz.open')
    @patch('pytesseract.image_to_data')
    @patch('extraction.tesseract_ocr_engine._get_tesseract_version')
    def test_ocr_pdf_single_page(self, mock_version, mock_image_to_data, mock_fitz_open):
        """Test OCR of single page PDF."""
        from pytesseract import Output
        
        mock_version.return_value = "5.0.0"
        
        # Mock PDF document
        mock_page = MagicMock()
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_pix = MagicMock()
        mock_pix.width = 612
        mock_pix.height = 792
        mock_pix.n = 3
        mock_pix.alpha = 0
        mock_pix.samples = bytes([255] * (612 * 792 * 3))
        mock_page.get_pixmap.return_value = mock_pix
        
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz_open.return_value = mock_doc
        
        # Mock Tesseract output
        mock_image_to_data.return_value = {
            "text": ["Test"],
            "left": [10],
            "top": [10],
            "width": [50],
            "height": [20],
            "conf": [90],
            "level": [5],
            "block_num": [1],
            "par_num": [1],
            "line_num": [1],
            "word_num": [1],
        }
        
        from extraction.tesseract_ocr_engine import ocr_pdf
        
        result = ocr_pdf("/path/to/test.pdf")
        
        assert len(result) == 1
        assert result[0].page_num == 1
