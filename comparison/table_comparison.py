"""Table structure and content comparison."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import re
from typing import Dict, List, Optional, Tuple

from comparison.alignment import align_pages
from comparison.models import Diff, PageData
from utils.coordinates import normalize_bbox
from utils.logging import logger
from utils.text_normalization import normalize_text


_NUM_TOKEN_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


@dataclass
class TableStructure:
    """Represents the structure of a table."""
    rows: int
    cols: int
    cells: List[List[str]]  # 2D array of cell contents
    bbox: Dict[str, float]  # Table bounding box
    confidence: float
    # Optional derived geometry to support lightweight style detection (padding/border weight)
    content_bbox: Optional[Dict[str, float]] = None
    content_insets_norm: Optional[Dict[str, float]] = None


def _table_text_signature(table: TableStructure, *, is_ocr_page: bool) -> str:
    """Build a normalized text signature for a table for quick style-only checks."""
    text = " ".join(" ".join(row) for row in (table.cells or []))
    return normalize_text(text, ocr=is_ocr_page)


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
                        "type": "table",
                        "table_change": "table_deleted",
                        "table_structure": {"rows": table.rows, "cols": table.cols},
                        "page_width": page_a.width,
                        "page_height": page_a.height,
                    },
                ))
            continue
        
        page_b = page_b_lookup[page_b_num]

        extraction_method_a = (page_a.metadata or {}).get("extraction_method", "")
        extraction_method_b = (page_b.metadata or {}).get("extraction_method", "")
        is_ocr_page = ("ocr" in (extraction_method_a or "").lower()) or ("ocr" in (extraction_method_b or "").lower())
        
        # Extract tables from both pages
        tables_a = _extract_tables_from_page(page_a)
        tables_b = _extract_tables_from_page(page_b)

        # Fallback: if layout detection missed a table on one side, try extracting
        # using the other side's bbox as a candidate region. This helps avoid
        # misclassifying table style tweaks (padding/border) as table added/removed.
        if tables_a and not tables_b:
            recovered_b: List[TableStructure] = []
            for ta in tables_a:
                rec = _extract_table_structure(page_b, {"bbox": ta.bbox, "confidence": ta.confidence})
                if rec:
                    recovered_b.append(rec)
            if recovered_b:
                tables_b = recovered_b

        if tables_b and not tables_a:
            recovered_a: List[TableStructure] = []
            for tb in tables_b:
                rec = _extract_table_structure(page_a, {"bbox": tb.bbox, "confidence": tb.confidence})
                if rec:
                    recovered_a.append(rec)
            if recovered_a:
                tables_a = recovered_a
        
        # Match tables between pages
        matched_pairs, unmatched_a, unmatched_b = _match_tables(tables_a, tables_b)
        
        # Compare matched tables
        for table_a, table_b in matched_pairs:
            table_diffs = _compare_table_structures(
                table_a,
                table_b,
                page_a.page_num,
                page_b.page_num,
                confidence,
                page_a.width,
                page_a.height,
                page_b.width,
                page_b.height,
                page_a,
                page_b,
                is_ocr_page=is_ocr_page,
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
                    "type": "table",
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
                    "type": "table",
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
            # Skip low-confidence detections (often false positives)
            try:
                from config.settings import settings

                conf = None
                if isinstance(table_region, dict):
                    conf = table_region.get("confidence")
                if conf is not None and float(conf) < settings.table_structure_confidence_threshold:
                    continue
            except Exception:
                # Best-effort filtering only
                pass

            # Try to extract table structure from PDF
            structure = _extract_table_structure(page, table_region)
            if structure:
                tables.append(structure)
    
    return _filter_contained_tables(tables)


def _filter_contained_tables(tables: List[TableStructure]) -> List[TableStructure]:
    """
    Filter out tables that strictly contain other tables (e.g. wrappers).
    Also filters duplicates.
    """
    if not tables:
        return []

    n = len(tables)
    to_remove = set()

    # Pre-calculate geometries
    geoms = []
    for t in tables:
        bx = t.bbox
        x, y = float(bx.get('x', 0)), float(bx.get('y', 0))
        w, h = float(bx.get('width', 0)), float(bx.get('height', 0))
        geoms.append((x, y, x + w, y + h, w * h))

    for i in range(n):
        if i in to_remove:
            continue

        xi1, yi1, xi2, yi2, area_i = geoms[i]

        for j in range(n):
            if i == j:
                continue
            if j in to_remove:
                continue

            xj1, yj1, xj2, yj2, area_j = geoms[j]

            # Intersection
            xx1 = max(xi1, xj1)
            yy1 = max(yi1, yj1)
            xx2 = min(xi2, xj2)
            yy2 = min(yi2, yj2)

            w_int = max(0.0, xx2 - xx1)
            h_int = max(0.0, yy2 - yy1)
            area_int = w_int * h_int

            if area_int <= 0:
                continue

            # Coverage relative to j (the candidate for removal if contained)
            coverage_j = area_int / area_j if area_j > 0 else 0.0
            
            # Coverage relative to i (the candidate for removal if duplicate)
            coverage_i = area_int / area_i if area_i > 0 else 0.0

            if coverage_j > 0.90:
                # j is mostly inside i
                if area_i > 1.2 * area_j:
                    # i is a container -> remove i (the larger wrapper)
                    to_remove.add(i)
                    break 
                else:
                    # Similar size -> Duplicate. Remove one (j).
                    to_remove.add(j)
            elif coverage_i > 0.90:
                 # i is mostly inside j
                 if area_j > 1.2 * area_i:
                     # j is container -> remove j
                     to_remove.add(j)

    return [t for k, t in enumerate(tables) if k not in to_remove]


def _extract_table_structure(page: PageData, table_region: dict) -> Optional[TableStructure]:
    """
    Extract table structure from a table region.
    
    This attempts to parse the table using PyMuPDF or camelot if available.
    Falls back to heuristic parsing if advanced tools are unavailable.
    """
    if importlib.util.find_spec("fitz") is None:
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

    # Track a tight content bbox for blocks that overlap the table region.
    content_x0: Optional[float] = None
    content_y0: Optional[float] = None
    content_x1: Optional[float] = None
    content_y1: Optional[float] = None
    
    for block in page.blocks:
        bx = block.bbox["x"]
        by = block.bbox["y"]
        bw = block.bbox["width"]
        bh = block.bbox["height"]
        bx1 = bx + bw
        by1 = by + bh
        
        # Check if block overlaps with table region
        if not (bx1 < x0 or bx > x1 or by1 < y0 or by > y1):
            # Expand the tight content bbox
            if content_x0 is None:
                content_x0, content_y0, content_x1, content_y1 = bx, by, bx1, by1
            else:
                content_x0 = min(content_x0, bx)
                content_y0 = min(content_y0, by)
                content_x1 = max(content_x1 or bx1, bx1)
                content_y1 = max(content_y1 or by1, by1)

            # Determine row (bin by y-coordinate)
            center_y = by + bh / 2
            row_idx = round((center_y - y0) / 20)  # Bin size of 20 points
            
            if row_idx not in row_blocks:
                row_blocks[row_idx] = []
            
            row_blocks[row_idx].append((bx, block.text))
    
    # Sort rows by y-coordinate
    sorted_rows = sorted(row_blocks.items(), key=lambda x: x[0])
    
    def _first_numeric_index(tokens: List[str]) -> Optional[int]:
        for i, token in enumerate(tokens):
            if _NUM_TOKEN_RE.match(token):
                return i
        return None

    for _, blocks_in_row in sorted_rows:
        # Sort blocks by x-coordinate (left to right)
        blocks_in_row.sort(key=lambda x: x[0])

        combined = " ".join(text for _, text in blocks_in_row).strip()
        if not combined:
            continue

        tokens = combined.split()
        if not tokens:
            continue

        numeric_start = _first_numeric_index(tokens)
        if numeric_start is not None and numeric_start > 0:
            # Data-like row: keep label (possibly multi-word) + numeric values.
            label = " ".join(tokens[:numeric_start]).strip()
            values = tokens[numeric_start:]
            row_cells = [label] + values
        else:
            # Header row or free text inside bbox: split into tokens.
            row_cells = tokens

        cells.append(row_cells)
        rows = len(cells)
        cols = max(cols, len(row_cells))
    
    if rows == 0 or cols == 0:
        return None
    
    # Pad rows to have same number of columns
    for row in cells:
        while len(row) < cols:
            row.append("")
    
    content_bbox: Optional[Dict[str, float]] = None
    content_insets_norm: Optional[Dict[str, float]] = None
    if content_x0 is not None and content_y0 is not None and content_x1 is not None and content_y1 is not None:
        content_bbox = {
            "x": float(content_x0),
            "y": float(content_y0),
            "width": float(max(0.0, content_x1 - content_x0)),
            "height": float(max(0.0, content_y1 - content_y0)),
        }

        table_w = float(bbox_dict.get("width", 0.0) or 0.0)
        table_h = float(bbox_dict.get("height", 0.0) or 0.0)
        if table_w > 1e-6 and table_h > 1e-6:
            left = (content_bbox["x"] - bbox_dict["x"]) / table_w
            top = (content_bbox["y"] - bbox_dict["y"]) / table_h
            right = ((bbox_dict["x"] + bbox_dict["width"]) - (content_bbox["x"] + content_bbox["width"])) / table_w
            bottom = ((bbox_dict["y"] + bbox_dict["height"]) - (content_bbox["y"] + content_bbox["height"])) / table_h

            # Clamp small negatives from float noise.
            content_insets_norm = {
                "left": max(-0.05, min(1.0, float(left))),
                "top": max(-0.05, min(1.0, float(top))),
                "right": max(-0.05, min(1.0, float(right))),
                "bottom": max(-0.05, min(1.0, float(bottom))),
            }

    return TableStructure(
        rows=rows,
        cols=cols,
        cells=cells,
        bbox=bbox_dict,
        confidence=table_region.get("confidence", 0.5),
        content_bbox=content_bbox,
        content_insets_norm=content_insets_norm,
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


def _bbox_area(b: Dict[str, float]) -> float:
    return float(b.get("width", 0.0)) * float(b.get("height", 0.0))


def _bbox_intersection_area(a: Dict[str, float], b: Dict[str, float]) -> float:
    ax0 = float(a.get("x", 0.0))
    ay0 = float(a.get("y", 0.0))
    ax1 = ax0 + float(a.get("width", 0.0))
    ay1 = ay0 + float(a.get("height", 0.0))

    bx0 = float(b.get("x", 0.0))
    by0 = float(b.get("y", 0.0))
    bx1 = bx0 + float(b.get("width", 0.0))
    by1 = by0 + float(b.get("height", 0.0))

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = ix1 - ix0
    ih = iy1 - iy0
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    return iw * ih


def _overlap_ratio(inner: Dict[str, float], outer: Dict[str, float]) -> float:
    """Return intersection area / inner area (0..1)."""
    a = _bbox_area(inner)
    if a <= 0.0:
        return 0.0
    return _bbox_intersection_area(inner, outer) / a


def _find_best_block_bbox_for_cell(
    page: PageData | None,
    *,
    table_bbox_abs: Dict[str, float],
    cell_text: str,
    is_ocr: bool,
) -> Optional[Dict[str, float]]:
    """
    Best-effort: locate a small bbox for a specific table cell by finding a page block
    whose text matches the cell content and overlaps the table region.

    This is critical to avoid highlighting the entire table when only one cell changes.
    """
    if page is None or not cell_text:
        return None

    target = normalize_text(cell_text, ocr=is_ocr).strip()
    if not target:
        return None

    best_bbox: Optional[Dict[str, float]] = None
    best_score: float = -1.0

    for block in (page.blocks or []):
        bb = getattr(block, "bbox", None)
        txt = getattr(block, "text", None)
        if not isinstance(bb, dict) or not txt:
            continue

        # Must meaningfully overlap the detected table region
        if _overlap_ratio(bb, table_bbox_abs) < 0.35:
            continue

        norm_block = normalize_text(txt, ocr=is_ocr).strip()
        if not norm_block:
            continue

        # Match heuristics:
        # - exact match is best
        # - containment helps for values inside longer OCR strings
        if norm_block == target:
            score = 10.0
        elif target in norm_block or norm_block in target:
            score = 5.0
        else:
            continue

        # Prefer smaller regions (cell-like), still within table bbox.
        area = _bbox_area(bb)
        if area > 0:
            score += 1.0 / (1.0 + area)

        if score > best_score:
            best_score = score
            best_bbox = bb

    return best_bbox


def _compare_table_structures(
    table_a: TableStructure,
    table_b: TableStructure,
    page_num_a: int,
    page_num_b: int,
    confidence: float,
    page_width_a: float,
    page_height_a: float,
    page_width_b: float,
    page_height_b: float,
    page_a: PageData | None = None,
    page_b: PageData | None = None,
    *,
    is_ocr_page: bool = False,
) -> List[Diff]:
    """Compare two matched tables and detect differences."""
    from config.settings import settings
    
    diffs: List[Diff] = []
    
    normalized_bbox_a = normalize_bbox(
        (table_a.bbox["x"], table_a.bbox["y"], 
         table_a.bbox["x"] + table_a.bbox["width"], 
         table_a.bbox["y"] + table_a.bbox["height"]),
        page_width_a, page_height_a
    )

    normalized_bbox_b = normalize_bbox(
        (table_b.bbox["x"], table_b.bbox["y"],
         table_b.bbox["x"] + table_b.bbox["width"],
         table_b.bbox["y"] + table_b.bbox["height"]),
        page_width_b, page_height_b
    )
    
    # =================================================================
    # OCR TABLE STRUCTURE COMPARISON ADJUSTMENT
    # OCR often segments table columns inconsistently due to spacing variance.
    # For OCR pages, we use more tolerant comparison:
    # - Only report structure change if ROWS differ (cols are unreliable)
    # - Or if column difference is >= 2 (not just ±1)
    # =================================================================
    rows_changed = table_a.rows != table_b.rows
    cols_changed = table_a.cols != table_b.cols
    
    if is_ocr_page:
        # For OCR: ignore small column differences (±1 col is likely OCR noise)
        col_diff = abs(table_b.cols - table_a.cols)
        structure_changed = rows_changed or col_diff >= 2

        # Heuristic: a single-column change can be real (e.g. an added "Delta" column)
        # when table detection is stable. Permit col_diff==1 if:
        # - table parsing confidence is decent on both sides
        # - table regions overlap strongly (stable bbox match)
        if (not structure_changed) and (not rows_changed) and col_diff == 1:
            table_confidence = min(table_a.confidence, table_b.confidence)
            overlap = _calculate_table_overlap(table_a, table_b)
            if table_confidence >= settings.ocr_table_structure_confidence_threshold and overlap >= 0.65:
                structure_changed = True
        
        # Log for debugging
        if cols_changed and col_diff < 2:
            logger.debug(
                "OCR table: ignoring small col diff (%dx%d vs %dx%d) on page %d",
                table_a.rows, table_a.cols, table_b.rows, table_b.cols, page_num_a
            )
    else:
        # For digital PDFs: strict comparison
        structure_changed = rows_changed or cols_changed
    
    if structure_changed:
        # Heuristic: if extracted table text is unchanged but our row/col parsing differs,
        # this is often a styling-only change (padding/border weight) that affects
        # block grouping inside the table region. Prefer a single formatting diff.
        if not is_ocr_page:
            sig_a = _table_text_signature(table_a, is_ocr_page=is_ocr_page)
            sig_b = _table_text_signature(table_b, is_ocr_page=is_ocr_page)
            if sig_a and sig_a == sig_b:
                diffs.append(
                    Diff(
                        page_num=page_num_a,
                        page_num_b=page_num_b,
                        diff_type="modified",
                        change_type="formatting",
                        old_text=None,
                        new_text=None,
                        bbox=normalized_bbox_a,
                        bbox_b=normalized_bbox_b,
                        confidence=confidence,
                        metadata={
                            "type": "table",
                            "table_change": "style",
                            "subtype": "structure_visibility",
                            "old_structure": {"rows": table_a.rows, "cols": table_a.cols},
                            "new_structure": {"rows": table_b.rows, "cols": table_b.cols},
                            "page_width": page_width_a,
                            "page_height": page_height_a,
                            "page_width_b": page_width_b,
                            "page_height_b": page_height_b,
                            "is_ocr": is_ocr_page,
                        },
                    )
                )
                return diffs

        row_change = table_b.rows - table_a.rows
        col_change = table_b.cols - table_a.cols

        # If only columns changed (common case: added “Delta” column), emit a single
        # content diff keyed by header change. This is both more informative and avoids
        # double-reporting when line_comparison also sees the table text.
        if row_change == 0 and col_change != 0:
            header_a = " ".join(table_a.cells[0]) if table_a.cells else f"Table ({table_a.rows}x{table_a.cols})"
            header_b = " ".join(table_b.cells[0]) if table_b.cells else f"Table ({table_b.rows}x{table_b.cols})"
            diffs.append(
                Diff(
                    page_num=page_num_a,
                    page_num_b=page_num_b,
                    diff_type="modified",
                    change_type="content",
                    old_text=header_a,
                    new_text=header_b,
                    bbox=normalized_bbox_a,
                    bbox_b=normalized_bbox_b,
                    confidence=confidence,
                    metadata={
                        "type": "table",
                        "table_change": "structure",
                        "subtype": "table_columns_changed",
                        "old_structure": {"rows": table_a.rows, "cols": table_a.cols},
                        "new_structure": {"rows": table_b.rows, "cols": table_b.cols},
                        "page_width": page_width_a,
                        "page_height": page_height_a,
                        "page_width_b": page_width_b,
                        "page_height_b": page_height_b,
                        "is_ocr": is_ocr_page,
                    },
                )
            )

            # For OCR pages with structure changes, don't do cell-level comparison
            # (too noisy when cols/rows are different)
            if is_ocr_page:
                return diffs
        else:
            diffs.append(Diff(
                page_num=page_num_a,
                page_num_b=page_num_b,
                diff_type="modified",
                change_type="layout",
                old_text=f"Table ({table_a.rows}x{table_a.cols})",
                new_text=f"Table ({table_b.rows}x{table_b.cols})",
                bbox=normalized_bbox_a,
                bbox_b=normalized_bbox_b,
                confidence=confidence,
                metadata={
                    "type": "table",
                    "table_change": "structure",
                    "old_structure": {"rows": table_a.rows, "cols": table_a.cols},
                    "new_structure": {"rows": table_b.rows, "cols": table_b.cols},
                    "page_width": page_width_a,
                    "page_height": page_height_a,
                    "page_width_b": page_width_b,
                    "page_height_b": page_height_b,
                    "is_ocr": is_ocr_page,
                },
            ))
        
        # For OCR pages with structure changes, don't do cell-level comparison
        # (too noisy when cols/rows are different)
        if is_ocr_page:
            return diffs
    
    # For OCR pages, check if table confidence is high enough for cell-level comparison
    if is_ocr_page:
        table_confidence = min(table_a.confidence, table_b.confidence)
        if table_confidence < settings.ocr_table_structure_confidence_threshold:
            # Confidence too low - just report region changed if texts differ significantly
            text_a = " ".join(" ".join(row) for row in table_a.cells)
            text_b = " ".join(" ".join(row) for row in table_b.cells)
            
            if normalize_text(text_a, ocr=True) != normalize_text(text_b, ocr=True):
                diffs.append(Diff(
                    page_num=page_num_a,
                    page_num_b=page_num_b,
                    diff_type="modified",
                    change_type="content",
                    old_text=f"Table content ({table_a.rows}x{table_a.cols})",
                    new_text=f"Table content ({table_b.rows}x{table_b.cols})",
                    bbox=normalized_bbox_a,
                    bbox_b=normalized_bbox_b,
                    confidence=table_confidence,
                    metadata={
                        "type": "table",
                        "table_change": "table_region_changed",
                        "low_confidence": True,
                        "needs_review": True,
                        "page_width": page_width_a,
                        "page_height": page_height_a,
                        "page_width_b": page_width_b,
                        "page_height_b": page_height_b,
                        "is_ocr": is_ocr_page,
                    },
                ))
            return diffs

    # -----------------------------------------------------------------
    # Lightweight table style detection (padding/border weight)
    # -----------------------------------------------------------------
    # If the table text is unchanged but either:
    # - content insets within the region shift, or
    # - the detected table region bbox shifts/sizes noticeably
    # then treat this as a likely table style change (padding/border weight).
    if not is_ocr_page:
        sig_a = _table_text_signature(table_a, is_ocr_page=is_ocr_page)
        sig_b = _table_text_signature(table_b, is_ocr_page=is_ocr_page)
        if sig_a and sig_a == sig_b:
            overlap = _calculate_table_overlap(table_a, table_b)
            if overlap >= 0.65:
                max_inset_delta = 0.0
                if table_a.content_insets_norm and table_b.content_insets_norm:
                    inset_keys = ("left", "top", "right", "bottom")
                    deltas = [
                        abs(
                            float(table_a.content_insets_norm.get(k, 0.0))
                            - float(table_b.content_insets_norm.get(k, 0.0))
                        )
                        for k in inset_keys
                    ]
                    max_inset_delta = max(deltas) if deltas else 0.0

                bbox_keys = ("x", "y", "width", "height")
                bbox_deltas = [
                    abs(float(normalized_bbox_a.get(k, 0.0)) - float(normalized_bbox_b.get(k, 0.0)))
                    for k in bbox_keys
                ]
                max_bbox_delta = max(bbox_deltas) if bbox_deltas else 0.0

                if (
                    max_inset_delta >= settings.table_style_inset_change_threshold
                    or max_bbox_delta >= settings.table_style_bbox_change_threshold
                ):
                    diffs.append(
                        Diff(
                            page_num=page_num_a,
                            page_num_b=page_num_b,
                            diff_type="modified",
                            change_type="formatting",
                            old_text=None,
                            new_text=None,
                            bbox=normalized_bbox_a,
                            bbox_b=normalized_bbox_b,
                            confidence=confidence,
                            metadata={
                                "type": "table",
                                "table_change": "style",
                                "subtype": "padding_or_border",
                                "overlap": overlap,
                                "insets_a": table_a.content_insets_norm,
                                "insets_b": table_b.content_insets_norm,
                                "max_inset_delta": max_inset_delta,
                                "max_bbox_delta": max_bbox_delta,
                                "page_width": page_width_a,
                                "page_height": page_height_a,
                                "page_width_b": page_width_b,
                                "page_height_b": page_height_b,
                                "is_ocr": is_ocr_page,
                            },
                        )
                    )
    
    # Compare cell contents (only if structure is same or digital PDF)
    max_rows = min(table_a.rows, table_b.rows)
    max_cols = min(table_a.cols, table_b.cols)
    
    for row_idx in range(max_rows):
        for col_idx in range(max_cols):
            cell_a = table_a.cells[row_idx][col_idx] if row_idx < len(table_a.cells) and col_idx < len(table_a.cells[row_idx]) else ""
            cell_b = table_b.cells[row_idx][col_idx] if row_idx < len(table_b.cells) and col_idx < len(table_b.cells[row_idx]) else ""
            
            # Use normalized comparison to ignore case and minor differences
            norm_a = normalize_text(cell_a, ocr=is_ocr_page)
            norm_b = normalize_text(cell_b, ocr=is_ocr_page)
            
            if norm_a != norm_b:
                # For OCR, check if the change is significant enough
                if is_ocr_page:
                    from utils.text_normalization import compute_ocr_change_significance
                    significance = compute_ocr_change_significance(cell_a, cell_b, ocr=True)
                    if not significance["is_significant"]:
                        continue  # Skip insignificant OCR noise

                # Prefer cell-level bbox (avoid highlighting entire table for single-cell changes)
                cell_bbox_a_abs = _find_best_block_bbox_for_cell(
                    page_a, table_bbox_abs=table_a.bbox, cell_text=cell_a, is_ocr=is_ocr_page
                )
                cell_bbox_b_abs = _find_best_block_bbox_for_cell(
                    page_b, table_bbox_abs=table_b.bbox, cell_text=cell_b, is_ocr=is_ocr_page
                )

                bbox_a = normalized_bbox_a
                if cell_bbox_a_abs:
                    bbox_a = normalize_bbox(
                        (
                            cell_bbox_a_abs["x"],
                            cell_bbox_a_abs["y"],
                            cell_bbox_a_abs["x"] + cell_bbox_a_abs["width"],
                            cell_bbox_a_abs["y"] + cell_bbox_a_abs["height"],
                        ),
                        page_width_a,
                        page_height_a,
                    )

                bbox_b = normalized_bbox_b
                if cell_bbox_b_abs:
                    bbox_b = normalize_bbox(
                        (
                            cell_bbox_b_abs["x"],
                            cell_bbox_b_abs["y"],
                            cell_bbox_b_abs["x"] + cell_bbox_b_abs["width"],
                            cell_bbox_b_abs["y"] + cell_bbox_b_abs["height"],
                        ),
                        page_width_b,
                        page_height_b,
                    )

                meta = {
                    "type": "table",
                    "table_change": "cell_content",
                    "row": row_idx,
                    "col": col_idx,
                    "page_width": page_width_a,
                    "page_height": page_height_a,
                    "page_width_b": page_width_b,
                    "page_height_b": page_height_b,
                    "is_ocr": is_ocr_page,
                    "cell_bbox_used": bool(cell_bbox_a_abs or cell_bbox_b_abs),
                }

                diffs.append(Diff(
                    page_num=page_num_a,
                    page_num_b=page_num_b,
                    diff_type="modified",
                    change_type="content",
                    old_text=cell_a,
                    new_text=cell_b,
                    bbox=bbox_a,
                    bbox_b=bbox_b,
                    confidence=confidence,
                    metadata=meta,
                ))
    
    return diffs

