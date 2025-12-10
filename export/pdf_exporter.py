"""Generate annotated PDF reports."""
from __future__ import annotations

from pathlib import Path
from typing import List

from comparison.models import ComparisonResult, Diff
from utils.logging import logger


def export_pdf(result: ComparisonResult, output_path: str | Path) -> Path:
    """
    Generate an annotated PDF report with highlighted differences.
    
    Args:
        result: ComparisonResult with diffs
        output_path: Path to save the output PDF
    
    Returns:
        Path to the generated PDF
    """
    output = Path(output_path)
    logger.info("Generating PDF report -> %s", output)
    
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for PDF export. Install via `pip install PyMuPDF`."
        ) from exc
    
    # Open the original document
    doc1_path = Path(result.doc1)
    if not doc1_path.exists():
        raise FileNotFoundError(f"Source document not found: {doc1_path}")
    
    doc = fitz.open(doc1_path)
    
    # Group diffs by page
    diffs_by_page: dict[int, List[Diff]] = {}
    for diff in result.diffs:
        if diff.page_num not in diffs_by_page:
            diffs_by_page[diff.page_num] = []
        diffs_by_page[diff.page_num].append(diff)
    
    # Add annotations to each page
    for page_num, diffs in diffs_by_page.items():
        if page_num > len(doc):
            continue
        
        page = doc[page_num - 1]  # 0-indexed
        
        for diff in diffs:
            if diff.bbox is None:
                continue
            
            # Get color based on diff type
            color_map = {
                "added": (0, 1, 0),      # Green
                "deleted": (1, 0, 0),    # Red
                "modified": (1, 0.84, 0),  # Gold
            }
            color = color_map.get(diff.diff_type, (0.5, 0.5, 0.5))
            
            # Get page dimensions for denormalization
            page_width = page.rect.width
            page_height = page.rect.height
            
            # Denormalize bbox from normalized dict format to absolute tuple coordinates
            denormalized_bbox = diff.denormalize_bbox(page_width, page_height)
            if denormalized_bbox is None:
                continue
            
            # denormalize_bbox already returns a tuple (x0, y0, x1, y1)
            x0, y0, x1, y1 = denormalized_bbox
            rect = fitz.Rect(x0, y0, x1, y1)
            
            # Add highlight annotation
            highlight = page.add_highlight_annot(rect)
            highlight.set_colors(stroke=color)
            highlight.set_opacity(0.3)
            highlight.update()
            
            # Add text note if available
            if diff.old_text or diff.new_text:
                note_text = f"{diff.diff_type.upper()}: {diff.change_type}\n"
                if diff.old_text:
                    note_text += f"Old: {diff.old_text[:100]}\n"
                if diff.new_text:
                    note_text += f"New: {diff.new_text[:100]}"
                
                # Add text annotation
                text_annot = page.add_text_annot(
                    fitz.Point(x0, y0),
                    note_text,
                )
                text_annot.set_colors(stroke=color)
                text_annot.update()
    
    # Add summary page at the beginning
    summary_page = doc.new_page(0)  # Insert at beginning
    _add_summary_page(summary_page, result)
    
    # Save the annotated PDF
    doc.save(output)
    doc.close()
    
    logger.info("PDF report generated: %s", output)
    return output


def _add_summary_page(page, result: ComparisonResult) -> None:
    """Add a summary page to the PDF report."""
    import fitz
    
    y_pos = 72
    line_height = 20
    
    # Title
    page.insert_text(
        (72, y_pos),
        "Document Comparison Report",
        fontsize=16,
        color=(0, 0, 0),
    )
    y_pos += line_height * 2
    
    # Document info
    page.insert_text(
        (72, y_pos),
        f"Document A: {Path(result.doc1).name}",
        fontsize=12,
    )
    y_pos += line_height
    
    page.insert_text(
        (72, y_pos),
        f"Document B: {Path(result.doc2).name}",
        fontsize=12,
    )
    y_pos += line_height * 2
    
    # Summary statistics - defensive handling for serialization issues
    summary = result.summary
    # Ensure summary is a dict, not a tuple or other type
    if not isinstance(summary, dict):
        logger.warning("Summary is not a dict (type: %s), creating default summary", type(summary))
        # Fallback: create summary from diffs
        from comparison.diff_classifier import get_diff_summary
        summary = get_diff_summary(result.diffs) if result.diffs else {"total": 0, "by_type": {}, "by_change_type": {}}
    
    page.insert_text(
        (72, y_pos),
        "Summary:",
        fontsize=14,
        color=(0, 0, 0),
    )
    y_pos += line_height
    
    total = summary.get('total', 0) if isinstance(summary, dict) else len(result.diffs) if hasattr(result, 'diffs') else 0
    page.insert_text(
        (72, y_pos),
        f"Total differences: {total}",
        fontsize=12,
    )
    y_pos += line_height
    
    # Breakdown by type
    if isinstance(summary, dict) and "by_type" in summary and isinstance(summary["by_type"], dict):
        page.insert_text(
            (72, y_pos),
            "By difference type:",
            fontsize=12,
            color=(0, 0, 0),
        )
        y_pos += line_height
        
        for diff_type, count in summary["by_type"].items():
            page.insert_text(
                (90, y_pos),
                f"  {diff_type}: {count}",
                fontsize=11,
            )
            y_pos += line_height
    
    # Breakdown by change type
    if isinstance(summary, dict) and "by_change_type" in summary and isinstance(summary["by_change_type"], dict):
        page.insert_text(
            (72, y_pos),
            "By change category:",
            fontsize=12,
            color=(0, 0, 0),
        )
        y_pos += line_height
        
        for change_type, count in summary["by_change_type"].items():
            page.insert_text(
                (90, y_pos),
                f"  {change_type}: {count}",
                fontsize=11,
            )
            y_pos += line_height
