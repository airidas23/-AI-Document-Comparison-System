"""Tests for visualization modules."""
from __future__ import annotations

import numpy as np
import pytest

from comparison.models import Diff
from utils.coordinates import bbox_dict_to_tuple, bbox_tuple_to_dict, denormalize_bbox, normalize_bbox
from utils.validation import validate_bbox_variety
from visualization.diff_renderer import COLOR_MAP, overlay_diffs, create_diff_summary_image


def test_overlay_diffs_noop():
    """Test overlay_diffs with no diffs."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = overlay_diffs(image, [], page_width=100, page_height=100)
    assert result.shape == image.shape


def test_overlay_diffs_with_bbox():
    """Test overlay_diffs with actual diff bounding boxes."""
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    diffs = [
        Diff(
            page_num=1,
            diff_type="added",
            change_type="content",
            old_text=None,
            new_text="New text",
            bbox={"x": 10, "y": 10, "width": 40, "height": 20},
            confidence=0.9,
        ),
    ]
    result = overlay_diffs(image, diffs, page_width=100, page_height=100, use_normalized=False)
    assert result.shape == image.shape
    # Image should be modified (not identical)
    assert not np.array_equal(result, image)


def test_overlay_diffs_with_normalized_bbox():
    """Test overlay_diffs with normalized bounding boxes (0-1 range)."""
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    # Normalized bbox: {"x": 0.1, "y": 0.1, "width": 0.4, "height": 0.2} corresponds to (10, 10, 50, 30) on 100x100 page
    diffs = [
        Diff(
            page_num=1,
            diff_type="added",
            change_type="content",
            old_text=None,
            new_text="New text",
            bbox={"x": 0.1, "y": 0.1, "width": 0.4, "height": 0.2},  # Normalized coordinates
            confidence=0.9,
        ),
    ]
    result = overlay_diffs(image, diffs, page_width=100, page_height=100, use_normalized=True)
    assert result.shape == image.shape
    # Image should be modified (not identical)
    assert not np.array_equal(result, image)


def test_overlay_diffs_without_cv2():
    """Test that overlay_diffs handles missing OpenCV gracefully."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    diffs = [
        Diff(
            page_num=1,
            diff_type="modified",
            change_type="content",
            old_text="Old",
            new_text="New",
            bbox={"x": 0, "y": 0, "width": 50, "height": 50},
        ),
    ]
    # Should not crash even if cv2 not available
    result = overlay_diffs(image, diffs, page_width=100, page_height=100)
    assert result is not None


def test_color_map():
    """Test that COLOR_MAP has expected keys."""
    assert "added" in COLOR_MAP
    assert "deleted" in COLOR_MAP
    assert "modified" in COLOR_MAP
    # Colors should be RGB tuples
    for color in COLOR_MAP.values():
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)


def test_create_diff_summary_image():
    """Test summary image creation."""
    diffs = [
        Diff(
            page_num=1,
            diff_type="added",
            change_type="content",
            old_text=None,
            new_text="Text",
            confidence=0.9,
        ),
    ]
    img = create_diff_summary_image(diffs, width=400, height=300)
    assert img.shape == (300, 400, 3)
    assert img.dtype == np.uint8


def test_normalize_bbox():
    """Test bbox coordinate normalization."""
    bbox = (100, 50, 200, 150)
    page_width = 1000
    page_height = 800
    
    normalized = normalize_bbox(bbox, page_width, page_height)
    
    # Check normalized values are in [0, 1] range and in dict format
    assert isinstance(normalized, dict)
    assert 0.0 <= normalized["x"] <= 1.0
    assert 0.0 <= normalized["y"] <= 1.0
    assert 0.0 <= normalized["width"] <= 1.0
    assert 0.0 <= normalized["height"] <= 1.0
    assert normalized["x"] == 0.1
    assert normalized["y"] == 0.0625
    assert normalized["width"] == 0.1  # (200-100)/1000
    assert normalized["height"] == 0.125  # (150-50)/800


def test_denormalize_bbox():
    """Test bbox coordinate denormalization."""
    normalized_bbox = {"x": 0.1, "y": 0.0625, "width": 0.1, "height": 0.125}
    page_width = 1000
    page_height = 800
    
    denormalized = denormalize_bbox(normalized_bbox, page_width, page_height)
    
    # Check denormalized values match original (as tuple)
    assert denormalized == (100.0, 50.0, 200.0, 150.0)


def test_normalize_denormalize_roundtrip():
    """Test that normalize -> denormalize returns original coordinates."""
    original_bbox = (100, 50, 200, 150)
    page_width = 1000
    page_height = 800
    
    normalized = normalize_bbox(original_bbox, page_width, page_height)
    denormalized = denormalize_bbox(normalized, page_width, page_height)
    
    assert denormalized == original_bbox


def test_normalize_bbox_clamping():
    """Test that normalize_bbox clamps coordinates to [0, 1] range."""
    # Bbox that extends beyond page boundaries
    bbox = (-50, -30, 1200, 900)
    page_width = 1000
    page_height = 800
    
    normalized = normalize_bbox(bbox, page_width, page_height)
    
    # All coordinates should be clamped to [0, 1]
    assert isinstance(normalized, dict)
    assert 0.0 <= normalized["x"] <= 1.0
    assert 0.0 <= normalized["y"] <= 1.0
    assert 0.0 <= normalized["width"] <= 1.0
    assert 0.0 <= normalized["height"] <= 1.0
    assert normalized["x"] == 0.0  # x clamped
    assert normalized["y"] == 0.0  # y clamped
    # Width and height should be adjusted to stay within bounds
    assert normalized["x"] + normalized["width"] <= 1.0
    assert normalized["y"] + normalized["height"] <= 1.0


def test_bbox_tuple_to_dict():
    """Test conversion from tuple to dict format."""
    bbox_tuple = (10, 20, 50, 60)
    bbox_dict = bbox_tuple_to_dict(bbox_tuple)
    
    assert bbox_dict == {"x": 10, "y": 20, "width": 40, "height": 40}


def test_bbox_dict_to_tuple():
    """Test conversion from dict to tuple format."""
    bbox_dict = {"x": 10, "y": 20, "width": 40, "height": 40}
    bbox_tuple = bbox_dict_to_tuple(bbox_dict)
    
    assert bbox_tuple == (10, 20, 50, 60)


def test_validate_bbox_variety():
    """Test bbox variety validation."""
    # Varied bboxes should pass
    varied_bboxes = [
        {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.1},
        {"x": 0.5, "y": 0.3, "width": 0.3, "height": 0.2},
        {"x": 0.2, "y": 0.6, "width": 0.4, "height": 0.15},
    ]
    assert validate_bbox_variety(varied_bboxes) is True
    
    # Identical bboxes should fail
    identical_bboxes = [
        {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.1},
        {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.1},
        {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.1},
    ]
    assert validate_bbox_variety(identical_bboxes) is False
    
    # Single bbox should pass (edge case)
    assert validate_bbox_variety([varied_bboxes[0]]) is True
    
    # Empty list should pass
    assert validate_bbox_variety([]) is True
