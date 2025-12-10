"""Convert diff data into visual highlights."""
from __future__ import annotations

from typing import List

import numpy as np

from comparison.models import Diff
from utils.logging import logger

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


# Color map matching React example: Green for additions, Red for deletions, Gold/Yellow for modifications
COLOR_MAP = {
    "added": (0, 255, 0),      # Green
    "deleted": (255, 0, 0),    # Red
    "modified": (255, 215, 0),  # Gold/Yellow
}

# Change type colors for visual distinction
CHANGE_TYPE_COLORS = {
    "content": (255, 215, 0),   # Gold for content changes
    "formatting": (255, 165, 0), # Orange for formatting changes
    "layout": (138, 43, 226),    # Blue violet for layout changes
    "visual": (255, 20, 147),    # Deep pink for visual changes
}


def overlay_diffs(
    image: np.ndarray,
    diffs: List[Diff],
    page_width: float,
    page_height: float,
    alpha: float = 0.3,
    use_normalized: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes and highlights for diffs on an image.
    
    Uses normalized coordinates (0-1) for responsive rendering, converting to
    pixel coordinates only at render time based on actual image size.
    
    Args:
        image: Input image (numpy array)
        diffs: List of Diff objects to highlight
        page_width: Original page width (for coordinate scaling)
        page_height: Original page height (for coordinate scaling)
        alpha: Transparency for highlights (0-1)
        use_normalized: If True, assumes bbox coordinates are normalized (0-1).
                       If False, treats bbox as absolute coordinates.
    
    Returns:
        Image with diff highlights overlaid
    """
    if cv2 is None:
        logger.warning("OpenCV not available, cannot render diff highlights")
        return image
    
    output = image.copy()
    img_height, img_width = image.shape[:2]
    
    logger.info("overlay_diffs called with %d diffs, image size: %dx%d", len(diffs), img_width, img_height)
    
    for i, diff in enumerate(diffs):
        if diff.bbox is None:
            continue
        
        # Get color based on diff type only (no blending to ensure consistent colors)
        # Green = added, Red = deleted, Yellow/Gold = modified
        color = COLOR_MAP.get(diff.diff_type, (128, 128, 128))
        
        # Handle normalized or absolute coordinates
        # bbox is in dict format: {"x": x, "y": y, "width": w, "height": h}
        if use_normalized:
            # bbox is in normalized coordinates (0-1), convert to pixel coordinates
            x = diff.bbox.get("x", 0.0)
            y = diff.bbox.get("y", 0.0)
            width = diff.bbox.get("width", 0.0)
            height = diff.bbox.get("height", 0.0)
            logger.debug("Diff %d: normalized bbox x=%.4f y=%.4f w=%.4f h=%.4f", i, x, y, width, height)
            img_x0 = int(x * img_width)
            img_y0 = int(y * img_height)
            img_x1 = int((x + width) * img_width)
            img_y1 = int((y + height) * img_height)
            logger.debug("Diff %d: pixel coords (%d,%d) to (%d,%d)", i, img_x0, img_y0, img_x1, img_y1)
        else:
            # bbox is in absolute coordinates, scale to image coordinates
            scale_x = img_width / page_width if page_width > 0 else 1.0
            scale_y = img_height / page_height if page_height > 0 else 1.0
            x = diff.bbox.get("x", 0.0)
            y = diff.bbox.get("y", 0.0)
            width = diff.bbox.get("width", 0.0)
            height = diff.bbox.get("height", 0.0)
            img_x0 = int(x * scale_x)
            img_y0 = int(y * scale_y)
            img_x1 = int((x + width) * scale_x)
            img_y1 = int((y + height) * scale_y)
        
        # Clamp to image bounds
        img_x0 = max(0, min(img_x0, img_width - 1))
        img_y0 = max(0, min(img_y0, img_height - 1))
        img_x1 = max(0, min(img_x1, img_width - 1))
        img_y1 = max(0, min(img_y1, img_height - 1))
        
        if img_x1 <= img_x0 or img_y1 <= img_y0:
            continue
        
        # Draw filled rectangle with transparency (matching React example style)
        overlay = output.copy()
        cv2.rectangle(
            overlay,
            (img_x0, img_y0),
            (img_x1, img_y1),
            color,
            thickness=-1,  # Filled
        )
        output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)
        
        # Draw border with slightly darker color for better visibility
        border_color = tuple(max(0, c - 30) for c in color)
        cv2.rectangle(
            output,
            (img_x0, img_y0),
            (img_x1, img_y1),
            border_color,
            thickness=2,
        )
    
    return output


def create_diff_summary_image(diffs: List[Diff], width: int = 800, height: int = 600) -> np.ndarray:
    """Create a summary visualization of all diffs."""
    if cv2 is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create blank image
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw legend
    y_offset = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    for diff_type, color in COLOR_MAP.items():
        # Draw color box
        cv2.rectangle(img, (20, y_offset - 15), (40, y_offset - 5), color, -1)
        # Draw label
        cv2.putText(
            img,
            f"{diff_type.capitalize()}: {sum(1 for d in diffs if d.diff_type == diff_type)}",
            (50, y_offset),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
        )
        y_offset += 25
    
    return img


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay a heatmap on an image with transparency.
    
    Args:
        image: Base image (numpy array, RGB)
        heatmap: Heatmap image (numpy array, RGB)
        alpha: Transparency factor (0-1), where 0 is fully transparent, 1 is fully opaque
    
    Returns:
        Image with heatmap overlaid
    """
    if cv2 is None:
        logger.warning("OpenCV not available, cannot overlay heatmap")
        return image
    
    # Resize heatmap to match image size if needed
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Ensure both are 3-channel images
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if len(heatmap.shape) == 2:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
    
    # Blend images
    result = cv2.addWeighted(image, 1.0 - alpha, heatmap, alpha, 0)
    
    return result
