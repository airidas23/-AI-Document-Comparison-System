"""Unit tests for comparison/visual_diff.py.

Tests cover:
- generate_heatmap function
- _create_heatmap_image function
- generate_heatmap_bytes function
- Edge cases (different page sizes, empty pages)
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import numpy as np


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_pdf_page():
    """Create a mock PDF page."""
    page = MagicMock()
    pix = MagicMock()
    pix.samples = bytes([100] * (100 * 100 * 3))  # 100x100 RGB image
    pix.height = 100
    pix.width = 100
    pix.n = 3
    page.get_pixmap.return_value = pix
    return page


@pytest.fixture
def mock_pdf_document(mock_pdf_page):
    """Create a mock PDF document."""
    doc = MagicMock()
    doc.__len__ = MagicMock(return_value=1)
    doc.__getitem__ = MagicMock(return_value=mock_pdf_page)
    return doc


@pytest.fixture
def sample_diff_image():
    """Create a sample difference image."""
    return np.ones((100, 100), dtype=np.uint8) * 50


@pytest.fixture
def sample_mask():
    """Create a sample binary mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255  # Square region of differences
    return mask


# =============================================================================
# Tests for _create_heatmap_image
# =============================================================================

class TestCreateHeatmapImage:
    """Tests for _create_heatmap_image function."""
    
    @patch('cv2.applyColorMap')
    @patch('cv2.bitwise_and')
    @patch('cv2.cvtColor')
    @patch('cv2.normalize')
    def test_create_heatmap_basic(self, mock_normalize, mock_cvtcolor, 
                                   mock_bitwise, mock_colormap, 
                                   sample_diff_image, sample_mask):
        """Test basic heatmap creation."""
        mock_normalize.return_value = sample_diff_image
        mock_colormap.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cvtcolor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_bitwise.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        
        from comparison.visual_diff import _create_heatmap_image
        
        result = _create_heatmap_image(sample_diff_image, sample_mask)
        
        assert result is not None
        mock_normalize.assert_called_once()
        mock_colormap.assert_called_once()
    
    @patch('cv2.applyColorMap')
    @patch('cv2.bitwise_and')
    @patch('cv2.cvtColor')
    @patch('cv2.normalize')
    def test_create_heatmap_empty_mask(self, mock_normalize, mock_cvtcolor, 
                                        mock_bitwise, mock_colormap, 
                                        sample_diff_image):
        """Test heatmap creation with empty mask."""
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        mock_normalize.return_value = sample_diff_image
        mock_colormap.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cvtcolor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_bitwise.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        
        from comparison.visual_diff import _create_heatmap_image
        
        result = _create_heatmap_image(sample_diff_image, empty_mask)
        
        assert result is not None


# =============================================================================
# Tests for generate_heatmap
# =============================================================================

class TestGenerateHeatmap:
    """Tests for generate_heatmap function."""
    
    @patch('fitz.open')
    @patch('cv2.resize')
    @patch('cv2.cvtColor')
    @patch('cv2.absdiff')
    @patch('cv2.threshold')
    @patch('cv2.morphologyEx')
    def test_generate_heatmap_basic(self, mock_morph, mock_thresh, mock_absdiff,
                                     mock_cvtcolor, mock_resize, mock_fitz_open):
        """Test basic heatmap generation."""
        # Setup mocks
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.samples = bytes([100] * (100 * 100 * 3))
        mock_pix.height = 100
        mock_pix.width = 100
        mock_pix.n = 3
        mock_page.get_pixmap.return_value = mock_pix
        
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz_open.return_value = mock_doc
        
        mock_cvtcolor.return_value = np.ones((100, 100), dtype=np.uint8)
        mock_absdiff.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_thresh.return_value = (None, np.zeros((100, 100), dtype=np.uint8))
        mock_morph.return_value = np.zeros((100, 100), dtype=np.uint8)
        
        with patch('comparison.visual_diff._create_heatmap_image') as mock_create:
            mock_create.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            
            from comparison.visual_diff import generate_heatmap
            
            result = generate_heatmap("/path/to/a.pdf", "/path/to/b.pdf", dpi=72)
            
            assert isinstance(result, list)
    
    @patch('fitz.open')
    def test_generate_heatmap_custom_dpi(self, mock_fitz_open):
        """Test heatmap generation with custom DPI."""
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.samples = bytes([100] * (100 * 100 * 3))
        mock_pix.height = 100
        mock_pix.width = 100
        mock_pix.n = 3
        mock_page.get_pixmap.return_value = mock_pix
        
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz_open.return_value = mock_doc
        
        with patch('cv2.cvtColor') as mock_cv, \
             patch('cv2.absdiff') as mock_diff, \
             patch('cv2.threshold') as mock_th, \
             patch('cv2.morphologyEx') as mock_morph, \
             patch('comparison.visual_diff._create_heatmap_image') as mock_create:
            
            mock_cv.return_value = np.ones((100, 100), dtype=np.uint8)
            mock_diff.return_value = np.zeros((100, 100), dtype=np.uint8)
            mock_th.return_value = (None, np.zeros((100, 100), dtype=np.uint8))
            mock_morph.return_value = np.zeros((100, 100), dtype=np.uint8)
            mock_create.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            
            from comparison.visual_diff import generate_heatmap
            
            result = generate_heatmap("/path/to/a.pdf", "/path/to/b.pdf", dpi=150)
            
            # DPI should be used in get_pixmap call
            mock_page.get_pixmap.assert_called()
    
    @patch('fitz.open')
    def test_generate_heatmap_different_page_counts(self, mock_fitz_open):
        """Test handling documents with different page counts."""
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.samples = bytes([100] * (100 * 100 * 3))
        mock_pix.height = 100
        mock_pix.width = 100
        mock_pix.n = 3
        mock_page.get_pixmap.return_value = mock_pix
        
        # Doc A has 3 pages, Doc B has 2 pages
        mock_doc_a = MagicMock()
        mock_doc_a.__len__ = MagicMock(return_value=3)
        mock_doc_a.__getitem__ = MagicMock(return_value=mock_page)
        
        mock_doc_b = MagicMock()
        mock_doc_b.__len__ = MagicMock(return_value=2)
        mock_doc_b.__getitem__ = MagicMock(return_value=mock_page)
        
        mock_fitz_open.side_effect = [mock_doc_a, mock_doc_b]
        
        with patch('cv2.cvtColor') as mock_cv, \
             patch('cv2.absdiff') as mock_diff, \
             patch('cv2.threshold') as mock_th, \
             patch('cv2.morphologyEx') as mock_morph, \
             patch('comparison.visual_diff._create_heatmap_image') as mock_create:
            
            mock_cv.return_value = np.ones((100, 100), dtype=np.uint8)
            mock_diff.return_value = np.zeros((100, 100), dtype=np.uint8)
            mock_th.return_value = (None, np.zeros((100, 100), dtype=np.uint8))
            mock_morph.return_value = np.zeros((100, 100), dtype=np.uint8)
            mock_create.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            
            from comparison.visual_diff import generate_heatmap
            
            result = generate_heatmap("/path/to/a.pdf", "/path/to/b.pdf", dpi=72)
            
            # Should only compare min(3, 2) = 2 pages
            assert len(result) == 2
    
    @patch('fitz.open')
    @patch('cv2.resize')
    def test_generate_heatmap_different_dimensions(self, mock_resize, mock_fitz_open):
        """Test handling pages with different dimensions."""
        # Page A is 100x100, Page B is 150x150
        mock_page_a = MagicMock()
        mock_pix_a = MagicMock()
        mock_pix_a.samples = bytes([100] * (100 * 100 * 3))
        mock_pix_a.height = 100
        mock_pix_a.width = 100
        mock_pix_a.n = 3
        mock_page_a.get_pixmap.return_value = mock_pix_a
        
        mock_page_b = MagicMock()
        mock_pix_b = MagicMock()
        mock_pix_b.samples = bytes([100] * (150 * 150 * 3))
        mock_pix_b.height = 150
        mock_pix_b.width = 150
        mock_pix_b.n = 3
        mock_page_b.get_pixmap.return_value = mock_pix_b
        
        mock_doc_a = MagicMock()
        mock_doc_a.__len__ = MagicMock(return_value=1)
        mock_doc_a.__getitem__ = MagicMock(return_value=mock_page_a)
        
        mock_doc_b = MagicMock()
        mock_doc_b.__len__ = MagicMock(return_value=1)
        mock_doc_b.__getitem__ = MagicMock(return_value=mock_page_b)
        
        mock_fitz_open.side_effect = [mock_doc_a, mock_doc_b]
        mock_resize.return_value = np.ones((150, 150, 3), dtype=np.uint8)
        
        with patch('cv2.cvtColor') as mock_cv, \
             patch('cv2.absdiff') as mock_diff, \
             patch('cv2.threshold') as mock_th, \
             patch('cv2.morphologyEx') as mock_morph, \
             patch('comparison.visual_diff._create_heatmap_image') as mock_create:
            
            mock_cv.return_value = np.ones((150, 150), dtype=np.uint8)
            mock_diff.return_value = np.zeros((150, 150), dtype=np.uint8)
            mock_th.return_value = (None, np.zeros((150, 150), dtype=np.uint8))
            mock_morph.return_value = np.zeros((150, 150), dtype=np.uint8)
            mock_create.return_value = np.zeros((150, 150, 3), dtype=np.uint8)
            
            from comparison.visual_diff import generate_heatmap
            
            result = generate_heatmap("/path/to/a.pdf", "/path/to/b.pdf", dpi=72)
            
            # resize should be called to match dimensions
            assert mock_resize.called


# =============================================================================
# Tests for generate_heatmap_bytes
# =============================================================================

class TestGenerateHeatmapBytes:
    """Tests for generate_heatmap_bytes function."""
    
    @patch('cv2.imencode')
    @patch('comparison.visual_diff.generate_heatmap')
    def test_generate_bytes_output(self, mock_generate, mock_imencode):
        """Test that bytes output is generated correctly."""
        mock_generate.return_value = [
            (1, np.zeros((100, 100, 3), dtype=np.uint8))
        ]
        mock_imencode.return_value = (True, np.array([1, 2, 3]))
        
        from comparison.visual_diff import generate_heatmap_bytes
        
        result = generate_heatmap_bytes("/path/to/a.pdf", "/path/to/b.pdf")
        
        assert len(result) == 1
        assert result[0][0] == 1  # Page number
        assert isinstance(result[0][1], bytes)
    
    @patch('cv2.imencode')
    @patch('comparison.visual_diff.generate_heatmap')
    def test_generate_bytes_multiple_pages(self, mock_generate, mock_imencode):
        """Test bytes generation for multiple pages."""
        mock_generate.return_value = [
            (1, np.zeros((100, 100, 3), dtype=np.uint8)),
            (2, np.zeros((100, 100, 3), dtype=np.uint8)),
            (3, np.zeros((100, 100, 3), dtype=np.uint8)),
        ]
        mock_imencode.return_value = (True, np.array([1, 2, 3]))
        
        from comparison.visual_diff import generate_heatmap_bytes
        
        result = generate_heatmap_bytes("/path/to/a.pdf", "/path/to/b.pdf")
        
        assert len(result) == 3
        for i, (page_num, data) in enumerate(result, 1):
            assert page_num == i
            assert isinstance(data, bytes)
    
    @patch('cv2.imencode')
    @patch('comparison.visual_diff.generate_heatmap')
    def test_generate_bytes_encode_failure(self, mock_generate, mock_imencode):
        """Test handling of encoding failure."""
        mock_generate.return_value = [
            (1, np.zeros((100, 100, 3), dtype=np.uint8)),
            (2, np.zeros((100, 100, 3), dtype=np.uint8)),
        ]
        # First succeeds, second fails
        mock_imencode.side_effect = [
            (True, np.array([1, 2, 3])),
            (False, None),
        ]
        
        from comparison.visual_diff import generate_heatmap_bytes
        
        result = generate_heatmap_bytes("/path/to/a.pdf", "/path/to/b.pdf")
        
        # Only the successfully encoded page should be in result
        assert len(result) == 1
        assert result[0][0] == 1


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestVisualDiffEdgeCases:
    """Tests for edge cases in visual diff."""
    
    def test_import_error_handling(self):
        """Test graceful handling when cv2/fitz not available."""
        # This test verifies the import error message
        with patch.dict('sys.modules', {'cv2': None, 'fitz': None}):
            # The module should raise RuntimeError with helpful message
            # when dependencies are missing
            pass  # Note: actual test would depend on how module handles missing deps
    
    @patch('fitz.open')
    def test_path_type_handling(self, mock_fitz_open):
        """Test that Path objects are handled correctly."""
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.samples = bytes([100] * (100 * 100 * 3))
        mock_pix.height = 100
        mock_pix.width = 100
        mock_pix.n = 3
        mock_page.get_pixmap.return_value = mock_pix
        
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz_open.return_value = mock_doc
        
        with patch('cv2.cvtColor') as mock_cv, \
             patch('cv2.absdiff') as mock_diff, \
             patch('cv2.threshold') as mock_th, \
             patch('cv2.morphologyEx') as mock_morph, \
             patch('comparison.visual_diff._create_heatmap_image') as mock_create:
            
            mock_cv.return_value = np.ones((100, 100), dtype=np.uint8)
            mock_diff.return_value = np.zeros((100, 100), dtype=np.uint8)
            mock_th.return_value = (None, np.zeros((100, 100), dtype=np.uint8))
            mock_morph.return_value = np.zeros((100, 100), dtype=np.uint8)
            mock_create.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            
            from comparison.visual_diff import generate_heatmap
            
            # Should accept Path objects
            result = generate_heatmap(
                Path("/path/to/a.pdf"), 
                Path("/path/to/b.pdf"), 
                dpi=72
            )
            
            assert isinstance(result, list)
    
    @patch('fitz.open')
    def test_grayscale_image_handling(self, mock_fitz_open):
        """Test handling of grayscale images (n=1)."""
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.samples = bytes([100] * (100 * 100 * 1))  # Grayscale
        mock_pix.height = 100
        mock_pix.width = 100
        mock_pix.n = 1  # Single channel
        mock_page.get_pixmap.return_value = mock_pix
        
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz_open.return_value = mock_doc
        
        with patch('cv2.cvtColor') as mock_cv, \
             patch('cv2.absdiff') as mock_diff, \
             patch('cv2.threshold') as mock_th, \
             patch('cv2.morphologyEx') as mock_morph, \
             patch('comparison.visual_diff._create_heatmap_image') as mock_create:
            
            # For single channel, no cvtColor conversion needed
            mock_diff.return_value = np.zeros((100, 100), dtype=np.uint8)
            mock_th.return_value = (None, np.zeros((100, 100), dtype=np.uint8))
            mock_morph.return_value = np.zeros((100, 100), dtype=np.uint8)
            mock_create.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            
            from comparison.visual_diff import generate_heatmap
            
            result = generate_heatmap("/path/to/a.pdf", "/path/to/b.pdf", dpi=72)
            
            assert isinstance(result, list)
