"""Cell-by-cell table comparison with column reordering detection."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
from comparison.models import Diff, PageData, TextBlock
from utils.coordinates import normalize_bbox
from utils.logging import logger
from utils.text_normalization import normalize_text


@dataclass
class TableCell:
    """Represents a single table cell."""
    row: int
    col: int
    text: str
    bbox: Optional[Dict[str, float]] = None


@dataclass
class ParsedTable:
    """Represents a parsed table structure."""
    rows: int
    cols: int
    cells: List[List[TableCell]]  # 2D array: cells[row][col]
    headers: List[str] = None  # Column headers if available
    bbox: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = []


def parse_table_structure(blocks: List[TextBlock], page_width: float, page_height: float) -> Optional[ParsedTable]:
    """
    Parse table structure from text blocks (Markdown or HTML format).
    
    Args:
        blocks: Text blocks that form a table
        page_width: Page width for coordinate normalization
        page_height: Page height for coordinate normalization
    
    Returns:
        ParsedTable if table structure is valid, None otherwise
    """
    if not blocks:
        return None
    
    # Try to parse as Markdown table first
    markdown_text = "\n".join(block.text for block in blocks)
    table = _parse_markdown_table(markdown_text, blocks, page_width, page_height)
    
    if table:
        return table
    
    # Try to parse as HTML table
    table = _parse_html_table(markdown_text, blocks, page_width, page_height)
    
    if table:
        return table
    
    # Fallback: heuristic parsing from text blocks
    return _parse_heuristic_table(blocks, page_width, page_height)


def _parse_markdown_table(
    text: str,
    blocks: List[TextBlock],
    page_width: float,
    page_height: float,
) -> Optional[ParsedTable]:
    """Parse Markdown table format (| col1 | col2 |)."""
    lines = text.strip().split('\n')
    table_rows: List[List[str]] = []
    
    for line in lines:
        line = line.strip()
        if not line.startswith('|'):
            continue
        
        # Remove leading/trailing |
        line = line.strip('|')
        # Split by | and clean
        cells = [cell.strip() for cell in line.split('|')]
        # Filter empty cells from edges
        cells = [c for c in cells if c]
        
        if cells:
            table_rows.append(cells)
    
    if len(table_rows) < 2:  # Need at least header and one data row
        return None
    
    # First row is usually header
    headers = table_rows[0]
    num_cols = len(headers)
    
    # Validate all rows have same number of columns
    for row in table_rows[1:]:
        if len(row) != num_cols:
            # Try to pad or truncate
            while len(row) < num_cols:
                row.append("")
            row = row[:num_cols]
    
    # Create cell structure
    cells: List[List[TableCell]] = []
    num_rows = len(table_rows)
    
    # Calculate approximate cell positions from blocks
    block_idx = 0
    for row_idx, row_data in enumerate(table_rows):
        row_cells: List[TableCell] = []
        for col_idx, cell_text in enumerate(row_data):
            # Try to find corresponding block bbox
            cell_bbox = None
            if block_idx < len(blocks):
                cell_bbox = blocks[block_idx].bbox
                block_idx += 1
            
            row_cells.append(TableCell(
                row=row_idx,
                col=col_idx,
                text=cell_text,
                bbox=cell_bbox,
            ))
        cells.append(row_cells)
    
    # Calculate table bbox from all cell bboxes
    table_bbox = _calculate_table_bbox(cells, page_width, page_height)
    
    return ParsedTable(
        rows=num_rows,
        cols=num_cols,
        cells=cells,
        headers=headers,
        bbox=table_bbox,
    )


def _parse_html_table(
    text: str,
    blocks: List[TextBlock],
    page_width: float,
    page_height: float,
) -> Optional[ParsedTable]:
    """Parse HTML table format."""
    try:
        soup = BeautifulSoup(text, 'html.parser')
        table_elem = soup.find('table')
        
        if not table_elem:
            return None
        
        rows_elem = table_elem.find_all('tr')
        if not rows_elem:
            return None
        
        # Parse rows
        cells: List[List[TableCell]] = []
        headers: List[str] = []
        block_idx = 0
        
        for row_idx, row_elem in enumerate(rows_elem):
            cell_elems = row_elem.find_all(['td', 'th'])
            if not cell_elems:
                continue
            
            row_cells: List[TableCell] = []
            for col_idx, cell_elem in enumerate(cell_elems):
                cell_text = cell_elem.get_text(strip=True)
                
                # Check if it's a header
                if row_idx == 0 and cell_elem.name == 'th':
                    headers.append(cell_text)
                
                # Get bbox if available
                cell_bbox = None
                if block_idx < len(blocks):
                    cell_bbox = blocks[block_idx].bbox
                    block_idx += 1
                
                row_cells.append(TableCell(
                    row=row_idx,
                    col=col_idx,
                    text=cell_text,
                    bbox=cell_bbox,
                ))
            
            cells.append(row_cells)
        
        if not cells:
            return None
        
        num_rows = len(cells)
        num_cols = max(len(row) for row in cells) if cells else 0
        
        # Pad rows to have same number of columns
        for row in cells:
            while len(row) < num_cols:
                row.append(TableCell(
                    row=row[0].row if row else 0,
                    col=len(row),
                    text="",
                    bbox=None,
                ))
        
        # If no headers extracted, use first row
        if not headers and cells:
            headers = [cell.text for cell in cells[0]]
        
        table_bbox = _calculate_table_bbox(cells, page_width, page_height)
        
        return ParsedTable(
            rows=num_rows,
            cols=num_cols,
            cells=cells,
            headers=headers,
            bbox=table_bbox,
        )
    except Exception as exc:
        logger.debug("HTML table parsing failed: %s", exc)
        return None


def _parse_heuristic_table(
    blocks: List[TextBlock],
    page_width: float,
    page_height: float,
) -> Optional[ParsedTable]:
    """Heuristic parsing: group blocks by rows based on y-coordinates."""
    if not blocks:
        return None
    
    # Group blocks by row (similar y-coordinates)
    row_groups: Dict[int, List[Tuple[float, TextBlock]]] = {}  # row_bin -> [(x, block)]
    
    for block in blocks:
        center_y = block.bbox["y"] + block.bbox["height"] / 2
        row_bin = round(center_y / 20)  # Bin size of 20 points
        
        if row_bin not in row_groups:
            row_groups[row_bin] = []
        
        center_x = block.bbox["x"] + block.bbox["width"] / 2
        row_groups[row_bin].append((center_x, block))
    
    # Sort rows by y-coordinate
    sorted_rows = sorted(row_groups.items(), key=lambda x: x[0])
    
    cells: List[List[TableCell]] = []
    for row_idx, (_, blocks_in_row) in enumerate(sorted_rows):
        # Sort blocks by x-coordinate (left to right)
        blocks_in_row.sort(key=lambda x: x[0])
        
        row_cells: List[TableCell] = []
        for col_idx, (_, block) in enumerate(blocks_in_row):
            row_cells.append(TableCell(
                row=row_idx,
                col=col_idx,
                text=block.text,
                bbox=block.bbox,
            ))
        cells.append(row_cells)
    
    if not cells:
        return None
    
    num_rows = len(cells)
    num_cols = max(len(row) for row in cells) if cells else 0
    
    # Pad rows
    for row in cells:
        while len(row) < num_cols:
            row.append(TableCell(
                row=row[0].row if row else 0,
                col=len(row),
                text="",
                bbox=None,
            ))
    
    # Use first row as headers
    headers = [cell.text for cell in cells[0]] if cells else []
    
    table_bbox = _calculate_table_bbox(cells, page_width, page_height)
    
    return ParsedTable(
        rows=num_rows,
        cols=num_cols,
        cells=cells,
        headers=headers,
        bbox=table_bbox,
    )


def _calculate_table_bbox(
    cells: List[List[TableCell]],
    page_width: float,
    page_height: float,
) -> Optional[Dict[str, float]]:
    """Calculate bounding box for entire table from cell bboxes."""
    if not cells or not cells[0]:
        return None
    
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    
    has_bbox = False
    for row in cells:
        for cell in row:
            if cell.bbox:
                min_x = min(min_x, cell.bbox["x"])
                min_y = min(min_y, cell.bbox["y"])
                max_x = max(max_x, cell.bbox["x"] + cell.bbox["width"])
                max_y = max(max_y, cell.bbox["y"] + cell.bbox["height"])
                has_bbox = True
    
    if not has_bbox:
        return None
    
    return {
        "x": min_x,
        "y": min_y,
        "width": max_x - min_x,
        "height": max_y - min_y,
    }


def compare_tables_cell_by_cell(
    table_a: ParsedTable,
    table_b: ParsedTable,
    page_num: int,
    page_width: float,
    page_height: float,
) -> List[Diff]:
    """
    Compare two tables cell-by-cell, handling column reordering.
    
    Args:
        table_a: First table
        table_b: Second table
        page_num: Page number for diffs
        page_width: Page width for coordinate normalization
        page_height: Page height for coordinate normalization
    
    Returns:
        List of Diff objects for cell changes
    """
    diffs: List[Diff] = []
    
    # Detect column reordering by matching headers
    col_mapping = _detect_column_mapping(table_a, table_b)
    
    # Normalize table bbox
    table_bbox_norm = None
    if table_a.bbox:
        table_bbox_norm = normalize_bbox(
            (table_a.bbox["x"], table_a.bbox["y"],
             table_a.bbox["x"] + table_a.bbox["width"],
             table_a.bbox["y"] + table_a.bbox["height"]),
            page_width,
            page_height,
        )
    
    # Compare structure changes
    if table_a.rows != table_b.rows or table_a.cols != table_b.cols:
        diffs.append(Diff(
            page_num=page_num,
            diff_type="modified",
            change_type="layout",
            old_text=f"Table structure: {table_a.rows}x{table_a.cols}",
            new_text=f"Table structure: {table_b.rows}x{table_b.cols}",
            bbox=table_bbox_norm,
            confidence=1.0,
            metadata={
                "table_change": "structure",
                "old_structure": {"rows": table_a.rows, "cols": table_a.cols},
                "new_structure": {"rows": table_b.rows, "cols": table_b.cols},
                "page_width": page_width,
                "page_height": page_height,
            },
        ))
    
    # Compare cells
    max_rows = min(table_a.rows, table_b.rows)
    max_cols = min(table_a.cols, table_b.cols)
    
    for row_idx in range(max_rows):
        for col_idx_a in range(max_cols):
            # Map column if reordered
            col_idx_b = col_mapping.get(col_idx_a, col_idx_a)
            
            if col_idx_b >= table_b.cols:
                # Column doesn't exist in table_b
                cell_a = table_a.cells[row_idx][col_idx_a] if row_idx < len(table_a.cells) and col_idx_a < len(table_a.cells[row_idx]) else None
                if cell_a:
                    cell_bbox_norm = None
                    if cell_a.bbox:
                        cell_bbox_norm = normalize_bbox(
                            (cell_a.bbox["x"], cell_a.bbox["y"],
                             cell_a.bbox["x"] + cell_a.bbox["width"],
                             cell_a.bbox["y"] + cell_a.bbox["height"]),
                            page_width,
                            page_height,
                        )
                    diffs.append(Diff(
                        page_num=page_num,
                        diff_type="deleted",
                        change_type="content",
                        old_text=cell_a.text,
                        new_text=None,
                        bbox=cell_bbox_norm or table_bbox_norm,
                        confidence=1.0,
                        metadata={
                            "table_change": "cell_deleted",
                            "row": row_idx,
                            "col": col_idx_a,
                            "page_width": page_width,
                            "page_height": page_height,
                        },
                    ))
                continue
            
            if col_idx_a >= table_a.cols:
                # Column doesn't exist in table_a
                cell_b = table_b.cells[row_idx][col_idx_b] if row_idx < len(table_b.cells) and col_idx_b < len(table_b.cells[row_idx]) else None
                if cell_b:
                    cell_bbox_norm = None
                    if cell_b.bbox:
                        cell_bbox_norm = normalize_bbox(
                            (cell_b.bbox["x"], cell_b.bbox["y"],
                             cell_b.bbox["x"] + cell_b.bbox["width"],
                             cell_b.bbox["y"] + cell_b.bbox["height"]),
                            page_width,
                            page_height,
                        )
                    diffs.append(Diff(
                        page_num=page_num,
                        diff_type="added",
                        change_type="content",
                        old_text=None,
                        new_text=cell_b.text,
                        bbox=cell_bbox_norm or table_bbox_norm,
                        confidence=1.0,
                        metadata={
                            "table_change": "cell_added",
                            "row": row_idx,
                            "col": col_idx_b,
                            "page_width": page_width,
                            "page_height": page_height,
                        },
                    ))
                continue
            
            # Compare cell content
            cell_a = table_a.cells[row_idx][col_idx_a] if row_idx < len(table_a.cells) and col_idx_a < len(table_a.cells[row_idx]) else None
            cell_b = table_b.cells[row_idx][col_idx_b] if row_idx < len(table_b.cells) and col_idx_b < len(table_b.cells[row_idx]) else None
            
            if not cell_a and not cell_b:
                continue
            
            text_a = normalize_text(cell_a.text) if cell_a else ""
            text_b = normalize_text(cell_b.text) if cell_b else ""
            
            if text_a != text_b:
                cell_bbox_norm = None
                if cell_a and cell_a.bbox:
                    cell_bbox_norm = normalize_bbox(
                        (cell_a.bbox["x"], cell_a.bbox["y"],
                         cell_a.bbox["x"] + cell_a.bbox["width"],
                         cell_a.bbox["y"] + cell_a.bbox["height"]),
                        page_width,
                        page_height,
                    )
                elif cell_b and cell_b.bbox:
                    cell_bbox_norm = normalize_bbox(
                        (cell_b.bbox["x"], cell_b.bbox["y"],
                         cell_b.bbox["x"] + cell_b.bbox["width"],
                         cell_b.bbox["y"] + cell_b.bbox["height"]),
                        page_width,
                        page_height,
                    )
                
                diffs.append(Diff(
                    page_num=page_num,
                    diff_type="modified",
                    change_type="content",
                    old_text=cell_a.text if cell_a else "",
                    new_text=cell_b.text if cell_b else "",
                    bbox=cell_bbox_norm or table_bbox_norm,
                    confidence=1.0,
                    metadata={
                        "table_change": "cell_content",
                        "row": row_idx,
                        "col_a": col_idx_a,
                        "col_b": col_idx_b,
                        "column_reordered": col_idx_a != col_idx_b,
                        "page_width": page_width,
                        "page_height": page_height,
                    },
                ))
    
    return diffs


def _detect_column_mapping(table_a: ParsedTable, table_b: ParsedTable) -> Dict[int, int]:
    """
    Detect column reordering by matching headers.
    
    Returns:
        Mapping from column index in table_a to column index in table_b
    """
    mapping: Dict[int, int] = {}
    used_b = set()
    
    # Normalize headers
    headers_a = [normalize_text(h) for h in table_a.headers] if table_a.headers else []
    headers_b = [normalize_text(h) for h in table_b.headers] if table_b.headers else []
    
    # If no headers, use first row
    if not headers_a and table_a.cells:
        headers_a = [normalize_text(cell.text) for cell in table_a.cells[0]]
    if not headers_b and table_b.cells:
        headers_b = [normalize_text(cell.text) for cell in table_b.cells[0]]
    
    # Match headers
    for col_idx_a, header_a in enumerate(headers_a):
        best_match = None
        best_similarity = 0.0
        
        for col_idx_b, header_b in enumerate(headers_b):
            if col_idx_b in used_b:
                continue
            
            # Calculate similarity
            words_a = set(header_a.split())
            words_b = set(header_b.split())
            
            if words_a and words_b:
                intersection = len(words_a & words_b)
                union = len(words_a | words_b)
                similarity = intersection / union if union > 0 else 0.0
            else:
                similarity = 1.0 if header_a == header_b else 0.0
            
            if similarity > best_similarity and similarity > 0.5:
                best_similarity = similarity
                best_match = col_idx_b
        
        if best_match is not None:
            mapping[col_idx_a] = best_match
            used_b.add(best_match)
        else:
            # No match found, use same index if available
            if col_idx_a < len(headers_b):
                mapping[col_idx_a] = col_idx_a
    
    # Fill in unmapped columns
    for col_idx_a in range(len(headers_a)):
        if col_idx_a not in mapping:
            # Find first unused column in table_b
            for col_idx_b in range(len(headers_b)):
                if col_idx_b not in used_b:
                    mapping[col_idx_a] = col_idx_b
                    used_b.add(col_idx_b)
                    break
    
    return mapping

