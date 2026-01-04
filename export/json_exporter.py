"""Export diff results as JSON."""
from __future__ import annotations

import json
from pathlib import Path

from comparison.models import ComparisonResult
from utils.logging import logger


def export_json(result: ComparisonResult, output_path: str | Path) -> Path:
    """
    Export comparison result as JSON with normalized coordinates.
    
    Includes normalized bbox coordinates (0-1) and page dimensions for
    coordinate denormalization when needed.
    """
    output = Path(output_path)
    logger.info("Writing JSON diff to %s", output)
    
    # Build diffs with explicit normalized coordinates and page dimensions
    diffs_data = []
    for diff in result.diffs:
        diff_dict = {
            "page_num": diff.page_num,
            "diff_type": diff.diff_type,
            "change_type": diff.change_type,
            "old_text": diff.old_text,
            "new_text": diff.new_text,
            "bbox": diff.bbox,  # Already normalized (0-1)
            "confidence": diff.confidence,
            "metadata": diff.metadata,
        }
        # Ensure page dimensions are in metadata for denormalization
        if "page_width" not in diff.metadata and diff.page_num:
            # Try to find page dimensions from result.pages
            for page in result.pages:
                if page.page_num == diff.page_num:
                    diff_dict["metadata"]["page_width"] = page.width
                    diff_dict["metadata"]["page_height"] = page.height
                    break
        diffs_data.append(diff_dict)
    
    payload = {
        "metadata": {"doc1": result.doc1, "doc2": result.doc2},
        "summary": result.summary,
        "pages": [
            {
                "page_num": page.page_num,
                "width": page.width,
                "height": page.height,
                "blocks": [
                    {
                        "text": block.text,
                        "bbox": block.bbox,  # Absolute coordinates (for reference)
                        "style": block.style.__dict__ if block.style else None
                    }
                    for block in page.blocks
                ],
                "lines": [
                    {
                        "line_id": line.line_id,
                        "text": line.text,
                        "bbox": line.bbox,
                        "confidence": line.confidence,
                        "reading_order": line.reading_order,
                        "tokens": [
                            {
                                "token_id": token.token_id,
                                "text": token.text,
                                "bbox": token.bbox,
                                "confidence": token.confidence,
                            }
                            for token in line.tokens
                        ],
                    }
                    for line in page.lines
                ],
            }
            for page in result.pages
        ],
        "diffs": diffs_data,
        "coordinate_system": {
            "bbox_format": "normalized_dict",
            "bbox_structure": {"x": "float", "y": "float", "width": "float", "height": "float"},
            "bbox_range": [0.0, 1.0],
            "description": (
                "Diff bboxes are normalized (0-1) in {x, y, width, height} format. "
                "Page blocks/lines/tokens use absolute PDF points in the same bbox dict format. "
                "Use page dimensions from metadata or pages array for denormalization."
            ),
        },
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
