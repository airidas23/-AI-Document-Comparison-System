"""Convert diff data into visual highlights."""
from __future__ import annotations

import re
from typing import List, Literal

import numpy as np

from comparison.models import Diff
from utils.logging import logger


_ALNUM_RE = re.compile(r"[A-Za-z0-9]")


def _is_approx_contained(inner: dict, outer: dict, tolerance: float = 0.01) -> bool:
    """
    Check if inner bbox is contained within outer bbox.
    
    Args:
        inner: Inner bbox dict {x, y, width, height}
        outer: Outer bbox dict {x, y, width, height}
        tolerance: Tolerance for floating point comparisons
        
    Returns:
        True if inner is strictly contained in or equal to outer
    """
    ix, iy, iw, ih = inner.get("x", 0), inner.get("y", 0), inner.get("width", 0), inner.get("height", 0)
    ox, oy, ow, oh = outer.get("x", 0), outer.get("y", 0), outer.get("width", 0), outer.get("height", 0)
    
    # Check bounds with tolerance
    # Inner Left >= Outer Left
    cond_x1 = ix >= (ox - tolerance)
    # Inner Top >= Outer Top
    cond_y1 = iy >= (oy - tolerance)
    # Inner Right <= Outer Right
    cond_x2 = (ix + iw) <= (ox + ow + tolerance)
    # Inner Bottom <= Outer Bottom
    cond_y2 = (iy + ih) <= (oy + oh + tolerance)
    
    return cond_x1 and cond_y1 and cond_x2 and cond_y2



def _is_punctuation_only(diff: Diff) -> bool:
    md = diff.metadata or {}
    if md.get("subtype") == "punctuation":
        return True

    old_text = (diff.old_text or "").strip()
    new_text = (diff.new_text or "").strip()
    if not old_text and not new_text:
        return False

    # If neither side has any alphanumeric characters, treat as punctuation/whitespace-only.
    return (not _ALNUM_RE.search(old_text)) and (not _ALNUM_RE.search(new_text))

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore


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


def _get_bboxes_to_render(
    diff: Diff,
    *,
    doc_side: Literal["a", "b"] = "a",
    use_word_bboxes: bool = True,
) -> List[dict]:
    """
    Get list of bboxes to render for a diff.
    
    Prefers word-level bboxes from metadata if available and use_word_bboxes=True,
    otherwise falls back to the diff's main bbox.
    
    Args:
        diff: Diff object with bbox and potentially word_bboxes in metadata
        use_word_bboxes: Whether to use word-level bboxes when available
    
    Returns:
        List of bbox dicts to render
    """
    # Render added/deleted diffs only on the relevant document side.
    # This prevents "red+green on both documents" when diffs don't have a side-specific bbox.
    if diff.diff_type == "added" and doc_side == "a":
        # If the diff explicitly carries B-side coordinates, it's meant for doc B only.
        if getattr(diff, "bbox_b", None) is not None or getattr(diff, "page_num_b", None) is not None:
            return []
    if diff.diff_type == "deleted" and doc_side == "b":
        return []

    md = diff.metadata or {}

    # For table diffs:
    # - If we have word-level bboxes (often from absorbed line/text diffs inside the table),
    #   prefer them to avoid rendering a huge "whole table" overlay *plus* the small cell box.
    # - Otherwise fall back to table bbox for true table-level events (added/deleted/style).
    if md.get("type") == "table":
        if use_word_bboxes:
            side_key = "word_bboxes_a" if doc_side == "a" else "word_bboxes_b"
            word_bboxes = md.get(side_key) or md.get("word_bboxes") or md.get("word_bboxes_all")
            if isinstance(word_bboxes, list) and word_bboxes:
                # Filter to dict bboxes only
                wb = [b for b in word_bboxes if isinstance(b, dict)]
                if wb:
                    return wb

        bbox = diff.bbox if doc_side == "a" else (getattr(diff, "bbox_b", None) or diff.bbox)
        if bbox is not None and isinstance(bbox, dict):
            return [bbox]
        return []

    if use_word_bboxes:
        # Prefer side-specific word bboxes (line_comparison provides *_a/*_b)
        side_key = "word_bboxes_a" if doc_side == "a" else "word_bboxes_b"
        word_bboxes = md.get(side_key)
        if not word_bboxes:
            # Back-compat / default: single list (often for doc A)
            word_bboxes = md.get("word_bboxes")
        if word_bboxes and isinstance(word_bboxes, list) and len(word_bboxes) > 0:
            return [b for b in word_bboxes if isinstance(b, dict)]
    
    # Fallback to main bbox (choose A vs B)
    bbox = diff.bbox if doc_side == "a" else (getattr(diff, "bbox_b", None) or diff.bbox)
    if bbox is not None and isinstance(bbox, dict):
        return [bbox]
    
    return []


def overlay_diffs(
    image: np.ndarray,
    diffs: List[Diff],
    page_width: float,
    page_height: float,
    alpha: float = 0.3,
    use_normalized: bool = True,
    use_word_bboxes: bool = True,
    doc_side: Literal["a", "b"] = "a",
) -> np.ndarray:
    """
    Draw bounding boxes and highlights for diffs on an image.
    
    Uses normalized coordinates (0-1) for responsive rendering, converting to
    pixel coordinates only at render time based on actual image size.
    
    Supports word-level highlighting when metadata["word_bboxes"] is available.
    
    Args:
        image: Input image (numpy array)
        diffs: List of Diff objects to highlight
        page_width: Original page width (for coordinate scaling)
        page_height: Original page height (for coordinate scaling)
        alpha: Transparency for highlights (0-1)
        use_normalized: If True, assumes bbox coordinates are normalized (0-1).
                       If False, treats bbox as absolute coordinates.
        use_word_bboxes: If True, use word-level bboxes from metadata when available.
    
    Returns:
        Image with diff highlights overlaid
    """
    output = image.copy()
    img_height, img_width = image.shape[:2]

    # Prevent tiny/degenerate highlights from appearing as dot/circle artifacts
    # (especially common with OCR token-level boxes in narrow table cells).
    min_box_px = 4
    
    logger.info("overlay_diffs called with %d diffs, image size: %dx%d", len(diffs), img_width, img_height)
    
    # First pass: Collect all render items
    render_items = []
    
    for i, diff in enumerate(diffs):

        # Skip rendering purely punctuation/whitespace-only diffs (keeps UI cleaner).
        # These can still appear in the diff list and summary.
        if diff.change_type == "formatting" and _is_punctuation_only(diff):
            continue

        # Get all bboxes to render (word-level or fallback to line-level)
        bboxes = _get_bboxes_to_render(diff, doc_side=doc_side, use_word_bboxes=use_word_bboxes)
        if not bboxes:
            continue
        
        # Get color based on diff type and change type
        # For modified diffs, use side-specific colors: deletions on doc A, additions on doc B.
        # This makes A/B overlays easier to interpret than using the same gold on both.
        if diff.diff_type == "modified":
            base_color = COLOR_MAP["deleted"] if doc_side == "a" else COLOR_MAP["added"]
        else:
            base_color = COLOR_MAP.get(diff.diff_type, (128, 128, 128))
        # Blend with change type color for better visual distinction
        change_color = CHANGE_TYPE_COLORS.get(diff.change_type, base_color)
        # Use 70% base color, 30% change type color
        color = tuple(
            int(base_color[j] * 0.7 + change_color[j] * 0.3)
            for j in range(3)
        )
        
        # Render each bbox (word-level granularity when available)
        for bbox in bboxes:
            # Handle normalized or absolute coordinates
            # bbox is in dict format: {"x": x, "y": y, "width": w, "height": h}
            if use_normalized:
                # bbox is in normalized coordinates (0-1), convert to pixel coordinates
                x = bbox.get("x", 0.0)
                y = bbox.get("y", 0.0)
                width = bbox.get("width", 0.0)
                height = bbox.get("height", 0.0)
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
                x = bbox.get("x", 0.0)
                y = bbox.get("y", 0.0)
                width = bbox.get("width", 0.0)
                height = bbox.get("height", 0.0)
                img_x0 = int(x * scale_x)
                img_y0 = int(y * scale_y)
                img_x1 = int((x + width) * scale_x)
                img_y1 = int((y + height) * scale_y)
            
            # Clamp to image bounds
            img_x0 = max(0, min(img_x0, img_width - 1))
            img_y0 = max(0, min(img_y0, img_height - 1))
            img_x1 = max(0, min(img_x1, img_width - 1))
            img_y1 = max(0, min(img_y1, img_height - 1))

            # Expand very small boxes to a minimum visible size while keeping center.
            bw = img_x1 - img_x0
            bh = img_y1 - img_y0
            if bw > 0 and bw < min_box_px:
                cx = (img_x0 + img_x1) / 2.0
                img_x0 = int(round(cx - min_box_px / 2.0))
                img_x1 = int(round(cx + min_box_px / 2.0))
            if bh > 0 and bh < min_box_px:
                cy = (img_y0 + img_y1) / 2.0
                img_y0 = int(round(cy - min_box_px / 2.0))
                img_y1 = int(round(cy + min_box_px / 2.0))

            img_x0 = max(0, min(img_x0, img_width - 1))
            img_y0 = max(0, min(img_y0, img_height - 1))
            img_x1 = max(0, min(img_x1, img_width - 1))
            img_y1 = max(0, min(img_y1, img_height - 1))
            
            if img_x1 <= img_x0 or img_y1 <= img_y0:
                continue
            
            # Store render item for deduplication
            render_items.append({
                "bbox_coords": (img_x0, img_y0, img_x1, img_y1),
                "original_bbox": bbox,
                "color": color,
                "area": (img_x1 - img_x0) * (img_y1 - img_y0)
            })

    # Deduplicate and remove containers
    # Logic: If box A contains box B, and box A is significantly larger, 
    # we assume box A is a "Group" highlighting and box B is "Detail".
    # Showing both obscures B. We hide A.
    # If box A == box B, we deduplicate.
    
    final_items = []
    # Sort items by area (smallest first) so we can efficiently check containment?
    # Actually, naive O(N^2) is fine for N < 1000 items.
    
    indices_to_skip = set()
    n_items = len(render_items)
    for j in range(n_items):
        if j in indices_to_skip:
            continue
            
        outer = render_items[j]
        ox0, oy0, ox1, oy1 = outer["bbox_coords"]
        
        is_redundant_container = False
        
        for k in range(n_items):
            if j == k: 
                continue
                
            inner = render_items[k]
            ix0, iy0, ix1, iy1 = inner["bbox_coords"]
            inner_area = inner["area"]
            
            # Calculate Intersection
            int_x0 = max(ox0, ix0)
            int_y0 = max(oy0, iy0)
            int_x1 = min(ox1, ix1)
            int_y1 = min(oy1, iy1)
            
            int_w = max(0, int_x1 - int_x0)
            int_h = max(0, int_y1 - int_y0)
            int_area = int_w * int_h
            
            # Check containment with fuzzy tolerance (IoU-like coverage)
            # If Inner is >90% covered by Outer
            is_covered = (inner_area > 0) and ((int_area / inner_area) > 0.90)
            
            if is_covered:
                # Outer covers Inner.
                # If they are effectively equal, treat as duplicate.
                if abs(outer["area"] - inner["area"]) < (outer["area"] * 0.1):
                    # Similar size -> Duplicate.
                    # Prefer keeping the first one, skip subsequent duplicates
                    if k > j:
                        indices_to_skip.add(k)
                    else:
                        # We are the duplicate of an earlier one
                        is_redundant_container = True
                        break
                else:
                    # Strict containment: Outer is significantly larger than Inner.
                    # Outer is likely a redundant container block (e.g. table wrapper).
                    # Hide Outer to show the detailed Inner diffs.
                    if outer["area"] > inner["area"]:
                        is_redundant_container = True
                        break
        
        if not is_redundant_container:
            final_items.append(outer)

    # Render filtered items
    if cv2 is not None:
        overlay = output.copy()
        
    for item in final_items:
        img_x0, img_y0, img_x1, img_y1 = item["bbox_coords"]
        color = item["color"]
        
        if cv2 is not None:
             # Draw filled rectangle with transparency (matching React example style)
            cv2.rectangle(
                overlay,
                (img_x0, img_y0),
                (img_x1, img_y1),
                color,
                thickness=-1,  # Filled
            )
            # Add to output later (batch addWeighted?) 
            # Original code did addWeighted per rect, which accumulates alpha.
            # Here we can draw all on one overlay and blend once? 
            # Or stick to original behavior? Original behavior accumulates alpha, making overlaps darker.
            # If we just removed overlaps, accumulation is less of an issue.
            # But let's stick to loop for safety. 
            pass

    if cv2 is not None:
        # Batch blending is safer to avoid alpha buildup saturation
        output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)
        
        # Draw borders on top
        for item in final_items:
            img_x0, img_y0, img_x1, img_y1 = item["bbox_coords"]
            color = item["color"]
            border_color = tuple(max(0, c - 30) for c in color)
            cv2.rectangle(
                output,
                (img_x0, img_y0),
                (img_x1, img_y1),
                border_color,
                thickness=2,
            )
    else:
        # PIL fallback
        if Image is None or ImageDraw is None:
             return output

        base = Image.fromarray(output.astype(np.uint8), mode="RGB").convert("RGBA")
        overlay_img = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_img)

        for item in final_items:
            img_x0, img_y0, img_x1, img_y1 = item["bbox_coords"]
            color = item["color"]
            
            fill = (color[0], color[1], color[2], int(255 * max(0.0, min(1.0, alpha))))
            border = (
                max(0, color[0] - 30),
                max(0, color[1] - 30),
                max(0, color[2] - 30),
                255,
            )
            draw.rectangle([img_x0, img_y0, img_x1, img_y1], fill=fill, outline=border, width=2)
            
        composed = Image.alpha_composite(base, overlay_img).convert("RGB")
        output = np.asarray(composed, dtype=np.uint8)
    
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
    if cv2 is not None:
        # Resize heatmap to match image size if needed
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        # Ensure both are 3-channel images
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if len(heatmap.shape) == 2:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)

        # Blend images
        return cv2.addWeighted(image, 1.0 - alpha, heatmap, alpha, 0)

    # PIL fallback
    if Image is None:
        logger.warning("Neither OpenCV nor PIL available; cannot overlay heatmap")
        return image

    base = Image.fromarray(image.astype(np.uint8), mode="RGB").convert("RGBA")
    hm = Image.fromarray(heatmap.astype(np.uint8), mode="RGB").convert("RGBA")
    if hm.size != base.size:
        hm = hm.resize(base.size)

    # Apply alpha to heatmap and composite
    hm.putalpha(int(255 * max(0.0, min(1.0, alpha))))
    return np.asarray(Image.alpha_composite(base, hm).convert("RGB"), dtype=np.uint8)
