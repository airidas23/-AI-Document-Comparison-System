"""Hierarchical alignment for document comparison using DTW and Needleman-Wunsch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from comparison.models import PageData, TextBlock
from utils.logging import logger
from utils.text_normalization import normalize_text


@dataclass
class DocumentSegment:
    """Represents a logical segment of a document (heading, paragraph, table, list)."""
    block_type: str  # "heading", "paragraph", "table", "list"
    heading_level: Optional[int] = None  # For headings: 1-6
    text: str = ""
    blocks: List[TextBlock] = None  # Text blocks in this segment
    parent_index: Optional[int] = None  # Index of parent segment (for hierarchical structure)
    children_indices: List[int] = None  # Indices of child segments
    
    def __post_init__(self):
        if self.blocks is None:
            self.blocks = []
        if self.children_indices is None:
            self.children_indices = []


def segment_document(page: PageData) -> List[DocumentSegment]:
    """
    Segment a document page into logical blocks using Markdown structure.
    
    Parses headings, paragraphs, tables, and lists from PageData metadata
    or infers structure from text patterns.
    
    Args:
        page: PageData with text blocks and metadata
    
    Returns:
        List of DocumentSegment objects in hierarchical order
    """
    segments: List[DocumentSegment] = []
    
    if not page.blocks:
        return segments
    
    # Check if we have Markdown structure in metadata
    has_markdown = page.metadata.get("has_markdown_structure", False)
    headings_metadata = page.metadata.get("headings", [])
    tables_metadata = page.metadata.get("tables", [])
    lists_metadata = page.metadata.get("lists", [])
    
    # Build index of special blocks
    heading_indices = {h["index"]: h for h in headings_metadata}
    table_row_indices = set()
    for table in tables_metadata:
        for row in table.get("rows", []):
            table_row_indices.add(row["index"])
    list_indices = {l["index"]: l for l in lists_metadata}
    
    # Track current heading hierarchy
    heading_stack: List[int] = []  # Stack of segment indices for headings
    
    current_paragraph_blocks: List[TextBlock] = []
    
    for idx, block in enumerate(page.blocks):
        # Check if this is a heading
        if idx in heading_indices:
            # Finalize current paragraph if any
            if current_paragraph_blocks:
                para_text = " ".join(b.text for b in current_paragraph_blocks)
                seg_idx = len(segments)
                segments.append(DocumentSegment(
                    block_type="paragraph",
                    text=para_text,
                    blocks=current_paragraph_blocks.copy(),
                    parent_index=heading_stack[-1] if heading_stack else None,
                ))
                if heading_stack:
                    parent_idx = heading_stack[-1]
                    segments[parent_idx].children_indices.append(seg_idx)
                current_paragraph_blocks = []
            
            # Create heading segment
            heading_info = heading_indices[idx]
            heading_seg = DocumentSegment(
                block_type="heading",
                heading_level=heading_info.get("level", 1),
                text=heading_info.get("text", block.text),
                blocks=[block],
                parent_index=heading_stack[-1] if heading_stack else None,
            )
            
            # Update heading stack - pop headings at same or higher level
            while heading_stack:
                parent_idx = heading_stack[-1]
                if segments[parent_idx].heading_level is None:
                    break
                if segments[parent_idx].heading_level >= heading_seg.heading_level:
                    heading_stack.pop()
                else:
                    break
            
            # Add to segments and update parent's children
            seg_idx = len(segments)
            segments.append(heading_seg)
            
            if heading_stack:
                parent_idx = heading_stack[-1]
                segments[parent_idx].children_indices.append(seg_idx)
            
            heading_stack.append(seg_idx)
            continue
        
        # Check if this is a table row
        if idx in table_row_indices:
            # Finalize current paragraph
            if current_paragraph_blocks:
                para_text = " ".join(b.text for b in current_paragraph_blocks)
                seg_idx = len(segments)
                segments.append(DocumentSegment(
                    block_type="paragraph",
                    text=para_text,
                    blocks=current_paragraph_blocks.copy(),
                    parent_index=heading_stack[-1] if heading_stack else None,
                ))
                if heading_stack:
                    parent_idx = heading_stack[-1]
                    segments[parent_idx].children_indices.append(seg_idx)
                current_paragraph_blocks = []
            
            # Find or create table segment
            table_seg_idx = None
            for seg_idx, seg in enumerate(segments):
                if seg.block_type == "table" and seg_idx == len(segments) - 1:
                    # Add to last table if it's the most recent
                    table_seg_idx = seg_idx
                    break
            
            if table_seg_idx is None:
                # Create new table segment
                table_seg = DocumentSegment(
                    block_type="table",
                    text="",
                    blocks=[],
                    parent_index=heading_stack[-1] if heading_stack else None,
                )
                table_seg_idx = len(segments)
                segments.append(table_seg)
                if heading_stack:
                    parent_idx = heading_stack[-1]
                    segments[parent_idx].children_indices.append(table_seg_idx)
            
            segments[table_seg_idx].blocks.append(block)
            segments[table_seg_idx].text += block.text + "\n"
            continue
        
        # Check if this is a list item
        if idx in list_indices:
            # Finalize current paragraph
            if current_paragraph_blocks:
                para_text = " ".join(b.text for b in current_paragraph_blocks)
                seg_idx = len(segments)
                segments.append(DocumentSegment(
                    block_type="paragraph",
                    text=para_text,
                    blocks=current_paragraph_blocks.copy(),
                    parent_index=heading_stack[-1] if heading_stack else None,
                ))
                if heading_stack:
                    parent_idx = heading_stack[-1]
                    segments[parent_idx].children_indices.append(seg_idx)
                current_paragraph_blocks = []
            
            # Create list item segment
            list_info = list_indices[idx]
            list_seg = DocumentSegment(
                block_type="list",
                text=list_info.get("text", block.text),
                blocks=[block],
                parent_index=heading_stack[-1] if heading_stack else None,
            )
            seg_idx = len(segments)
            segments.append(list_seg)
            if heading_stack:
                parent_idx = heading_stack[-1]
                segments[parent_idx].children_indices.append(seg_idx)
            continue
        
        # Regular paragraph block
        current_paragraph_blocks.append(block)
    
    # Finalize last paragraph
    if current_paragraph_blocks:
        para_text = " ".join(b.text for b in current_paragraph_blocks)
        seg_idx = len(segments)
        segments.append(DocumentSegment(
            block_type="paragraph",
            text=para_text,
            blocks=current_paragraph_blocks.copy(),
            parent_index=heading_stack[-1] if heading_stack else None,
        ))
        if heading_stack:
            parent_idx = heading_stack[-1]
            segments[parent_idx].children_indices.append(seg_idx)
    
    logger.debug("Segmented page into %d logical blocks", len(segments))
    return segments


def hierarchical_align(
    page_a: PageData,
    page_b: PageData,
    use_dtw: bool = True,
) -> Dict[int, Tuple[int, float]]:
    """
    Align two pages hierarchically using section headings and paragraph alignment.
    
    First aligns section headings, then aligns paragraphs within each section.
    
    Args:
        page_a: First page
        page_b: Second page
        use_dtw: If True, use DTW for paragraph alignment; otherwise use Needleman-Wunsch
    
    Returns:
        Mapping from segment index in page_a to (segment index in page_b, confidence)
    """
    segments_a = segment_document(page_a)
    segments_b = segment_document(page_b)
    
    if not segments_a or not segments_b:
        # Fallback to simple block alignment
        return {}
    
    alignment_map: Dict[int, Tuple[int, float]] = {}
    
    # First level: Align headings
    heading_segments_a = [(i, seg) for i, seg in enumerate(segments_a) if seg.block_type == "heading"]
    heading_segments_b = [(i, seg) for i, seg in enumerate(segments_b) if seg.block_type == "heading"]
    
    heading_alignment = _align_headings(heading_segments_a, heading_segments_b)
    
    # Second level: Align paragraphs within each section
    for heading_idx_a, (heading_idx_b, confidence) in heading_alignment.items():
        # Get all segments under this heading in doc_a
        section_a = _get_section_segments(segments_a, heading_idx_a)
        section_b = _get_section_segments(segments_b, heading_idx_b)
        
        # Align paragraphs in this section
        if use_dtw:
            para_alignment = align_with_dtw(section_a, section_b)
        else:
            para_alignment = align_with_needleman_wunsch(section_a, section_b)
        
        # Merge alignments
        for seg_idx_a, (seg_idx_b, conf) in para_alignment.items():
            alignment_map[seg_idx_a] = (seg_idx_b, min(confidence, conf))
    
    # Handle unmatched headings and segments
    matched_b = set(seg_idx_b for (seg_idx_b, _) in alignment_map.values())
    for seg_idx_a, seg_a in enumerate(segments_a):
        if seg_idx_a not in alignment_map:
            # Try to find best match in unmatched segments_b
            best_match, best_conf = _find_best_segment_match(seg_a, segments_b, matched_b)
            if best_match is not None and best_conf > 0.3:
                alignment_map[seg_idx_a] = (best_match, best_conf)
                matched_b.add(best_match)
    
    logger.debug("Hierarchical alignment: %d segment mappings", len(alignment_map))
    return alignment_map


def _align_headings(
    headings_a: List[Tuple[int, DocumentSegment]],
    headings_b: List[Tuple[int, DocumentSegment]],
) -> Dict[int, Tuple[int, float]]:
    """Align headings between two documents using semantic similarity."""
    alignment: Dict[int, Tuple[int, float]] = {}
    matched_b = set()
    
    for idx_a, seg_a in headings_a:
        best_match = None
        best_confidence = 0.0
        
        for idx_b, seg_b in headings_b:
            if idx_b in matched_b:
                continue
            
            # Check heading level match
            level_match = seg_a.heading_level == seg_b.heading_level
            
            # Calculate text similarity
            text_a = normalize_text(seg_a.text)
            text_b = normalize_text(seg_b.text)
            
            # Simple word overlap similarity
            words_a = set(text_a.split())
            words_b = set(text_b.split())
            
            if words_a and words_b:
                intersection = len(words_a & words_b)
                union = len(words_a | words_b)
                similarity = intersection / union if union > 0 else 0.0
            else:
                similarity = 0.0
            
            # Combine level match and text similarity
            confidence = similarity * 0.7 + (0.3 if level_match else 0.0)
            
            if confidence > best_confidence and confidence > 0.4:
                best_confidence = confidence
                best_match = idx_b
        
        if best_match is not None:
            alignment[idx_a] = (best_match, best_confidence)
            matched_b.add(best_match)
    
    return alignment


def _get_section_segments(segments: List[DocumentSegment], heading_idx: int) -> List[Tuple[int, DocumentSegment]]:
    """Get all segments that belong to a section (under a heading)."""
    section: List[Tuple[int, DocumentSegment]] = []
    heading = segments[heading_idx]
    
    # Get direct children
    for child_idx in heading.children_indices:
        if child_idx < len(segments):
            section.append((child_idx, segments[child_idx]))
    
    # Also include segments that come after this heading until next heading of same or higher level
    if heading.heading_level is not None:
        for idx in range(heading_idx + 1, len(segments)):
            seg = segments[idx]
            if seg.block_type == "heading" and seg.heading_level is not None:
                if seg.heading_level <= heading.heading_level:
                    break
            section.append((idx, seg))
    
    return section


def align_with_dtw(
    segments_a: List[Tuple[int, DocumentSegment]],
    segments_b: List[Tuple[int, DocumentSegment]],
) -> Dict[int, Tuple[int, float]]:
    """
    Align segments using Dynamic Time Warping (DTW).
    
    Args:
        segments_a: List of (index, segment) tuples from first document
        segments_b: List of (index, segment) tuples from second document
    
    Returns:
        Mapping from segment index in doc_a to (segment index in doc_b, confidence)
    """
    try:
        from dtaidistance import dtw
    except ImportError:
        logger.warning("dtaidistance not available, falling back to simple alignment")
        return _simple_segment_alignment(segments_a, segments_b)
    
    if not segments_a or not segments_b:
        return {}
    
    # Create sequences of normalized text for comparison
    seq_a = [normalize_text(seg.text) for _, seg in segments_a]
    seq_b = [normalize_text(seg.text) for _, seg in segments_b]
    
    # Convert to distance matrix using simple word overlap
    def distance_func(text1: str, text2: str) -> float:
        words1 = set(text1.split()) if text1 else set()
        words2 = set(text2.split()) if text2 else set()
        if not words1 and not words2:
            return 0.0
        if not words1 or not words2:
            return 1.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return 1.0 - (intersection / union if union > 0 else 0.0)
    
    # Compute DTW distance matrix
    distance_matrix = [[distance_func(seq_a[i], seq_b[j]) for j in range(len(seq_b))] 
                       for i in range(len(seq_a))]
    
    # Compute DTW path
    try:
        d, path = dtw.warping_path(distance_matrix)
        
        # Build alignment map from path
        alignment_map: Dict[int, Tuple[int, float]] = {}
        for i, j in path:
            idx_a = segments_a[i][0]
            idx_b = segments_b[j][0]
            
            # Calculate confidence from distance
            dist = distance_matrix[i][j]
            confidence = max(0.0, 1.0 - dist)
            
            if idx_a not in alignment_map or confidence > alignment_map[idx_a][1]:
                alignment_map[idx_a] = (idx_b, confidence)
        
        return alignment_map
    except Exception as exc:
        logger.warning("DTW computation failed: %s, using simple alignment", exc)
        return _simple_segment_alignment(segments_a, segments_b)


def align_with_needleman_wunsch(
    segments_a: List[Tuple[int, DocumentSegment]],
    segments_b: List[Tuple[int, DocumentSegment]],
    match_score: float = 1.0,
    mismatch_penalty: float = -1.0,
    gap_penalty: float = -0.5,
) -> Dict[int, Tuple[int, float]]:
    """
    Align segments using modified Needleman-Wunsch algorithm.
    
    Args:
        segments_a: List of (index, segment) tuples from first document
        segments_b: List of (index, segment) tuples from second document
        match_score: Score for matching segments
        mismatch_penalty: Penalty for mismatched segments
        gap_penalty: Penalty for gaps (insertions/deletions)
    
    Returns:
        Mapping from segment index in doc_a to (segment index in doc_b, confidence)
    """
    if not segments_a or not segments_b:
        return {}
    
    n = len(segments_a)
    m = len(segments_b)
    
    # Initialize scoring matrix
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    
    # Initialize first row and column (gap penalties)
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + gap_penalty
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + gap_penalty
    
    # Fill scoring matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            seg_a = segments_a[i - 1][1]
            seg_b = segments_b[j - 1][1]
            
            # Calculate similarity
            text_a = normalize_text(seg_a.text)
            text_b = normalize_text(seg_b.text)
            words_a = set(text_a.split())
            words_b = set(text_b.split())
            
            if words_a and words_b:
                intersection = len(words_a & words_b)
                union = len(words_a | words_b)
                similarity = intersection / union if union > 0 else 0.0
            else:
                similarity = 0.0
            
            # Match/mismatch score
            match = match_score if similarity > 0.5 else mismatch_penalty
            
            # Three options: match, gap in a, gap in b
            score[i][j] = max(
                score[i - 1][j - 1] + match,
                score[i - 1][j] + gap_penalty,
                score[i][j - 1] + gap_penalty,
            )
    
    # Traceback to find alignment
    alignment_map: Dict[int, Tuple[int, float]] = {}
    i, j = n, m
    
    while i > 0 and j > 0:
        seg_a_idx = segments_a[i - 1][0]
        seg_b_idx = segments_b[j - 1][0]
        
        # Calculate similarity for confidence
        text_a = normalize_text(segments_a[i - 1][1].text)
        text_b = normalize_text(segments_b[j - 1][1].text)
        words_a = set(text_a.split())
        words_b = set(text_b.split())
        
        if words_a and words_b:
            intersection = len(words_a & words_b)
            union = len(words_a | words_b)
            confidence = intersection / union if union > 0 else 0.0
        else:
            confidence = 0.0
        
        # Determine which direction to go
        if score[i][j] == score[i - 1][j - 1] + (match_score if confidence > 0.5 else mismatch_penalty):
            # Match
            alignment_map[seg_a_idx] = (seg_b_idx, confidence)
            i -= 1
            j -= 1
        elif score[i][j] == score[i - 1][j] + gap_penalty:
            # Gap in a (deletion)
            i -= 1
        else:
            # Gap in b (insertion)
            j -= 1
    
    return alignment_map


def _simple_segment_alignment(
    segments_a: List[Tuple[int, DocumentSegment]],
    segments_b: List[Tuple[int, DocumentSegment]],
) -> Dict[int, Tuple[int, float]]:
    """Simple alignment based on position and text similarity (fallback)."""
    alignment_map: Dict[int, Tuple[int, float]] = {}
    matched_b = set()
    
    for idx_a, seg_a in segments_a:
        best_match = None
        best_confidence = 0.0
        
        for idx_b, seg_b in segments_b:
            if idx_b in matched_b:
                continue
            
            # Calculate text similarity
            text_a = normalize_text(seg_a.text)
            text_b = normalize_text(seg_b.text)
            words_a = set(text_a.split())
            words_b = set(text_b.split())
            
            if words_a and words_b:
                intersection = len(words_a & words_b)
                union = len(words_a | words_b)
                similarity = intersection / union if union > 0 else 0.0
            else:
                similarity = 0.0
            
            if similarity > best_confidence and similarity > 0.3:
                best_confidence = similarity
                best_match = idx_b
        
        if best_match is not None:
            alignment_map[idx_a] = (best_match, best_confidence)
            matched_b.add(best_match)
    
    return alignment_map


def _find_best_segment_match(
    segment: DocumentSegment,
    segments_b: List[DocumentSegment],
    exclude_indices: set,
) -> Tuple[Optional[int], float]:
    """Find best matching segment in segments_b for given segment."""
    best_match = None
    best_confidence = 0.0
    
    text_a = normalize_text(segment.text)
    words_a = set(text_a.split())
    
    for idx, seg_b in enumerate(segments_b):
        if idx in exclude_indices:
            continue
        
        text_b = normalize_text(seg_b.text)
        words_b = set(text_b.split())
        
        if words_a and words_b:
            intersection = len(words_a & words_b)
            union = len(words_a | words_b)
            similarity = intersection / union if union > 0 else 0.0
        else:
            similarity = 0.0
        
        if similarity > best_confidence:
            best_confidence = similarity
            best_match = idx
    
    return (best_match, best_confidence)

