"""Table structure and content comparison."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from comparison.alignment import align_pages
from comparison.models import Diff, PageData
from utils.coordinates import normalize_bbox
from utils.logging import logger
from utils.text_normalization import normalize_text


@dataclass
class TableStructure:
    """Represents the structure of a table."""
    rows: int
    cols: int
    cells: List[List[str]]  # 2D array of cell contents
    bbox: Dict[str, float]  # Table bounding box
    confidence: float


def compare_tables(
    pages_a: List[PageData],
    pages_b: List[PageData],
    alignment_map: dict | None = None,
) -> List[Diff]:
    """
    Compare tables between two documents.
    
    Detects:
    - Table additions/deletions
    - Table structure changes (added/removed rows/columns)
    - Cell content changes
    
    Args:
        pages_a: Pages from first document
        pages_b: Pages from second document
        alignment_map: Optional pre-computed page alignment
    
    Returns:
        List of Diff objects representing table changes
    """
    logger.info("Comparing tables between documents")
    
    if alignment_map is None:
        alignment_map = align_pages(pages_a, pages_b, use_similarity=False)
    
    all_diffs: List[Diff] = []
    page_b_lookup = {page.page_num: page for page in pages_b}
    
    for page_a in pages_a:
        if page_a.page_num not in alignment_map:
            continue
        
        page_b_num, confidence = alignment_map[page_a.page_num]
        if page_b_num not in page_b_lookup:
            # All tables in page_a are deleted
            tables_a = _extract_tables_from_page(page_a)
            for table in tables_a:
                normalized_bbox = normalize_bbox(
                    (table.bbox["x"], table.bbox["y"], 
                     table.bbox["x"] + table.bbox["width"], 
                     table.bbox["y"] + table.bbox["height"]),
                    page_a.width, page_a.height
                )
                all_diffs.append(Diff(
                    page_num=page_a.page_num,
                    diff_type="deleted",
                    change_type="layout",
                    old_text=f"Table ({table.rows}x{table.cols})",
                    new_text=None,
                    bbox=normalized_bbox,
                    confidence=confidence,
                    metadata={
                        "table_change": "table_deleted",
                        "table_structure": {"rows": table.rows, "cols": table.cols},
                        "page_width": page_a.width,
                        "page_height": page_a.height,
                    },
                ))
            continue
        
        page_b = page_b_lookup[page_b_num]
        
        # Extract tables from both pages
        tables_a = _extract_tables_from_page(page_a)
        tables_b = _extract_tables_from_page(page_b)
        
        # Match tables between pages
        matched_pairs, unmatched_a, unmatched_b = _match_tables(tables_a, tables_b)
        
        # Compare matched tables
        for table_a, table_b in matched_pairs:
            table_diffs = _compare_table_structures(
                table_a, table_b, page_a.page_num, confidence,
                page_a.width, page_a.height
            )
            all_diffs.extend(table_diffs)
        
        # Unmatched tables in doc_a are deleted
        for table in unmatched_a:
            normalized_bbox = normalize_bbox(
                (table.bbox["x"], table.bbox["y"], 
                 table.bbox["x"] + table.bbox["width"], 
                 table.bbox["y"] + table.bbox["height"]),
                page_a.width, page_a.height
            )
            all_diffs.append(Diff(
                page_num=page_a.page_num,
                diff_type="deleted",
                change_type="layout",
                old_text=f"Table ({table.rows}x{table.cols})",
                new_text=None,
                bbox=normalized_bbox,
                confidence=confidence,
                metadata={
                    "table_change": "table_deleted",
                    "table_structure": {"rows": table.rows, "cols": table.cols},
                    "page_width": page_a.width,
                    "page_height": page_a.height,
                },
            ))
        
        # Unmatched tables in doc_b are added
        for table in unmatched_b:
            normalized_bbox = normalize_bbox(
                (table.bbox["x"], table.bbox["y"], 
                 table.bbox["x"] + table.bbox["width"], 
                 table.bbox["y"] + table.bbox["height"]),
                page_b.width, page_b.height
            )
            all_diffs.append(Diff(
                page_num=page_b.page_num,
                diff_type="added",
                change_type="layout",
                old_text=None,
                new_text=f"Table ({table.rows}x{table.cols})",
                bbox=normalized_bbox,
                confidence=confidence,
                metadata={
                    "table_change": "table_added",
                    "table_structure": {"rows": table.rows, "cols": table.cols},
                    "page_width": page_b.width,
                    "page_height": page_b.height,
                },
            ))
    
    logger.info("Detected %d table differences", len(all_diffs))
    return all_diffs


def _extract_tables_from_page(page: PageData) -> List[TableStructure]:
    """Extract table structures from a page."""
    tables = []
    
    # Get tables from page metadata if available
    if "tables" in page.metadata:
        table_regions = page.metadata.get("tables", [])
        for table_region in table_regions:
            # Try to extract table structure from PDF
            structure = _extract_table_structure(page, table_region)
            if structure:
                tables.append(structure)
    
    return tables


def _extract_table_structure(page: PageData, table_region: dict) -> Optional[TableStructure]:
    """
    Extract table structure from a table region.
    
    This attempts to parse the table using PyMuPDF or camelot if available.
    Falls back to heuristic parsing if advanced tools are unavailable.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.debug("PyMuPDF not available for table extraction")
        return None
    
    # Handle table_region - it might be a dict or the bbox directly
    if not isinstance(table_region, dict):
        logger.debug("table_region is not a dict: %s", type(table_region))
        return None
    
    bbox = table_region.get("bbox")
    if not bbox:
        logger.debug("No bbox in table_region")
        return None
    
    # Convert bbox to dict format if needed
    # bbox can be: list, tuple, or dict
    if isinstance(bbox, (list, tuple)):
        if len(bbox) < 4:
            logger.debug("bbox has insufficient elements: %d", len(bbox))
            return None
        try:
            # Handle both [x0, y0, x1, y1] and (x0, y0, x1, y1) formats
            x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            bbox_dict = {
                "x": x0,
                "y": y0,
                "width": x1 - x0,
                "height": y1 - y0,
            }
        except (ValueError, TypeError, IndexError) as exc:
            logger.debug("Error parsing bbox tuple/list: %s", exc)
            return None
    elif isinstance(bbox, dict):
        # Already a dict, just ensure it has the right format
        try:
            bbox_dict = {
                "x": float(bbox.get("x", 0)),
                "y": float(bbox.get("y", 0)),
                "width": float(bbox.get("width", 0)),
                "height": float(bbox.get("height", 0)),
            }
        except (ValueError, TypeError) as exc:
            logger.debug("Error parsing bbox dict: %s", exc)
            return None
    else:
        logger.debug("Unexpected bbox type: %s, value: %s", type(bbox), bbox)
        return None
    
    # Try to extract table using PyMuPDF's table detection
    # For now, use a simple heuristic to extract table structure
    # In production, this would use Table Transformer or camelot
    
    # Find text blocks within table region
    cells = []
    rows = 0
    cols = 0
    
    x0 = bbox_dict["x"]
    y0 = bbox_dict["y"]
    x1 = x0 + bbox_dict["width"]
    y1 = y0 + bbox_dict["height"]
    
    # Group blocks by rows (similar y-coordinates)
    row_blocks: Dict[int, List[Tuple[float, str]]] = {}  # row_index -> [(x, text)]
    
    for block in page.blocks:
        bx = block.bbox["x"]
        by = block.bbox["y"]
        bw = block.bbox["width"]
        bh = block.bbox["height"]
        bx1 = bx + bw
        by1 = by + bh
        
        # Check if block overlaps with table region
        if not (bx1 < x0 or bx > x1 or by1 < y0 or by > y1):
            # Determine row (bin by y-coordinate)
            center_y = by + bh / 2
            row_idx = round((center_y - y0) / 20)  # Bin size of 20 points
            
            if row_idx not in row_blocks:
                row_blocks[row_idx] = []
            
            row_blocks[row_idx].append((bx, block.text))
    
    # Sort rows by y-coordinate
    sorted_rows = sorted(row_blocks.items(), key=lambda x: x[0])
    
    for row_idx, (_, blocks_in_row) in enumerate(sorted_rows):
        # Sort blocks by x-coordinate (left to right)
        blocks_in_row.sort(key=lambda x: x[0])
        
        row_cells = [text for _, text in blocks_in_row]
        if row_cells:
            cells.append(row_cells)
            rows = max(rows, len(cells))
            cols = max(cols, len(row_cells))
    
    if rows == 0 or cols == 0:
        return None
    
    # Pad rows to have same number of columns
    for row in cells:
        while len(row) < cols:
            row.append("")
    
    return TableStructure(
        rows=rows,
        cols=cols,
        cells=cells,
        bbox=bbox_dict,
        confidence=table_region.get("confidence", 0.5),
    )


def _match_tables(
    tables_a: List[TableStructure],
    tables_b: List[TableStructure],
) -> Tuple[List[Tuple[TableStructure, TableStructure]], List[TableStructure], List[TableStructure]]:
    """
    Match tables between two pages based on position and structure similarity.
    
    Returns:
        Tuple of (matched_pairs, unmatched_a, unmatched_b)
    """
    matched: List[Tuple[TableStructure, TableStructure]] = []
    unmatched_a = tables_a.copy()
    unmatched_b = tables_b.copy()
    
    # Simple matching: match tables by position overlap
    for table_a in tables_a:
        best_match = None
        best_score = 0.0
        
        for table_b in unmatched_b:
            # Calculate overlap score
            from config.settings import settings
            score = _calculate_table_overlap(table_a, table_b)
            if score > best_score and score > settings.table_overlap_threshold:
                best_score = score
                best_match = table_b
        
        if best_match:
            matched.append((table_a, best_match))
            unmatched_a.remove(table_a)
            unmatched_b.remove(best_match)
    
    return matched, unmatched_a, unmatched_b


def _calculate_table_overlap(table_a: TableStructure, table_b: TableStructure) -> float:
    """Calculate overlap score between two tables based on position."""
    bbox_a = table_a.bbox
    bbox_b = table_b.bbox
    
    # Calculate intersection area
    x0 = max(bbox_a["x"], bbox_b["x"])
    y0 = max(bbox_a["y"], bbox_b["y"])
    x1 = min(bbox_a["x"] + bbox_a["width"], bbox_b["x"] + bbox_b["width"])
    y1 = min(bbox_a["y"] + bbox_a["height"], bbox_b["y"] + bbox_b["height"])
    
    if x1 <= x0 or y1 <= y0:
        return 0.0
    
    intersection = (x1 - x0) * (y1 - y0)
    area_a = bbox_a["width"] * bbox_a["height"]
    area_b = bbox_b["width"] * bbox_b["height"]
    union = area_a + area_b - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def _compare_table_structures(
    table_a: TableStructure,
    table_b: TableStructure,
    page_num: int,
    confidence: float,
    page_width: float,
    page_height: float,
) -> List[Diff]:
    """Compare two matched tables and detect differences."""
    diffs: List[Diff] = []
    
    normalized_bbox = normalize_bbox(
        (table_a.bbox["x"], table_a.bbox["y"], 
         table_a.bbox["x"] + table_a.bbox["width"], 
         table_a.bbox["y"] + table_a.bbox["height"]),
        page_width, page_height
    )
    
    # Check for structure changes
    if table_a.rows != table_b.rows or table_a.cols != table_b.cols:
        row_change = table_b.rows - table_a.rows
        col_change = table_b.cols - table_a.cols
        
        change_desc = []
        if row_change != 0:
            change_desc.append(f"{abs(row_change)} row{'s' if abs(row_change) > 1 else ''} {'added' if row_change > 0 else 'removed'}")
        if col_change != 0:
            change_desc.append(f"{abs(col_change)} column{'s' if abs(col_change) > 1 else ''} {'added' if col_change > 0 else 'removed'}")
        
        diffs.append(Diff(
            page_num=page_num,
            diff_type="modified",
            change_type="layout",
            old_text=f"Table ({table_a.rows}x{table_a.cols})",
            new_text=f"Table ({table_b.rows}x{table_b.cols})",
            bbox=normalized_bbox,
            confidence=confidence,
            metadata={
                "table_change": "structure",
                "old_structure": {"rows": table_a.rows, "cols": table_a.cols},
                "new_structure": {"rows": table_b.rows, "cols": table_b.cols},
                "page_width": page_width,
                "page_height": page_height,
            },
        ))
    
    # Compare cell contents
    max_rows = min(table_a.rows, table_b.rows)
    max_cols = min(table_a.cols, table_b.cols)
    
    for row_idx in range(max_rows):
        for col_idx in range(max_cols):
            cell_a = table_a.cells[row_idx][col_idx] if row_idx < len(table_a.cells) and col_idx < len(table_a.cells[row_idx]) else ""
            cell_b = table_b.cells[row_idx][col_idx] if row_idx < len(table_b.cells) and col_idx < len(table_b.cells[row_idx]) else ""
            
            # Use normalized comparison to ignore case and minor differences
            if normalize_text(cell_a) != normalize_text(cell_b):
                diffs.append(Diff(
                    page_num=page_num,
                    diff_type="modified",
                    change_type="content",
                    old_text=cell_a,
                    new_text=cell_b,
                    bbox=normalized_bbox,  # Use table bbox for now
                    confidence=confidence,
                    metadata={
                        "table_change": "cell_content",
                        "row": row_idx,
                        "col": col_idx,
                        "page_width": page_width,
                        "page_height": page_height,
                    },
                ))
    
    return diffs

