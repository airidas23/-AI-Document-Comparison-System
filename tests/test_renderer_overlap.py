import numpy as np
import pytest
from comparison.models import Diff
from visualization.diff_renderer import overlay_diffs

def test_renderer_hides_container_bbox():
    """
    Test that if one diff completely contains another, the container is hidden.
    """
    # Create 100x100 white image
    # Use RGB format simulation
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    # Outer diff covering (10,10) to (90,90) = 80x80
    diff_outer = Diff(
        page_num=1,
        diff_type="deleted",
        change_type="content",
        bbox={"x": 10, "y": 10, "width": 80, "height": 80},
        old_text="Outer",
        new_text=None
    )
    
    # Inner diff covering (40,40) to (60,60) = 20x20
    diff_inner = Diff(
        page_num=1,
        diff_type="deleted",
        change_type="content",
        bbox={"x": 40, "y": 40, "width": 20, "height": 20},
        old_text="Inner",
        new_text=None
    )
    
    # Render with alpha=1.0 (opaque) to verify cleaning
    # use_normalized=False
    output = overlay_diffs(
        image, 
        [diff_outer, diff_inner], 
        page_width=100, 
        page_height=100, 
        alpha=1.0, 
        use_normalized=False, 
        use_word_bboxes=False
    )
    
    # Pixel at (20,20) is inside Outer but outside Inner.
    # It SHOULD be White (255) if Outer is removed.
    # It WOULD be colored if Outer was drawn.
    p_outer_region = output[20, 20]
    
    # Pixel at (50,50) is inside Inner.
    # It SHOULD be colored.
    p_inner_region = output[50, 50]
    
    print(f"Outer region pixel: {p_outer_region}")
    print(f"Inner region pixel: {p_inner_region}")
    
    # Check Outer is hidden
    assert np.all(p_outer_region == [255, 255, 255]), "Container box should be hidden/removed"
    
    # Check Inner is drawn (not white)
    assert not np.all(p_inner_region == [255, 255, 255]), "Inner box should be visible"

def test_renderer_keeps_non_overlapping():
    """Test that non-overlapping boxes are both kept."""
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    diff_1 = Diff(
        page_num=1,
        diff_type="deleted",
        change_type="content",
        bbox={"x": 10, "y": 10, "width": 10, "height": 10},
        old_text="Box1", new_text=None
    )
    diff_2 = Diff(
        page_num=1,
        diff_type="deleted",
        change_type="content",
        bbox={"x": 30, "y": 30, "width": 10, "height": 10},
        old_text="Box2", new_text=None
    )
    
    output = overlay_diffs(
        image, 
        [diff_1, diff_2], 
        page_width=100, 
        page_height=100, 
        alpha=1.0, 
        use_normalized=False, 
        use_word_bboxes=False
    )
    
    # Both should be colored
    assert not np.all(output[15, 15] == [255, 255, 255])
    assert not np.all(output[35, 35] == [255, 255, 255])
