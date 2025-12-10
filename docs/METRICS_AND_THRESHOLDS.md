# Metrics and Thresholds Reference

This document lists all metrics and thresholds extracted from the PDF Document Comparison AI Prototype Plan and implemented in the system.

## Performance Metrics

### Target Performance
- **Seconds per page target**: 3.0 seconds (as specified in PDF)
- **Maximum pages**: 60 pages per document
- **Total time target**: ~180 seconds (3 minutes) for a 60-page document pair

### Performance Tracking
All performance metrics are tracked via `utils/performance.py`:
- Time per page calculation
- Total processing time
- Performance target compliance checking

## Change Detection Metrics

### Target Metrics (from PDF requirements)
- **Precision**: ≥ 85% (target_precision = 0.85)
- **Recall**: ≥ 80% (target_recall = 0.80)
- **F1 Score**: ≥ 82% (target_f1_score = 0.82)
- **Alignment Accuracy**: ≥ 95% (target_alignment_accuracy = 0.95)

### Evaluation Metrics Available
See `utils/metrics.py` for implementation:
- Precision, Recall, F1 Score, Accuracy
- Change type breakdown (content, formatting, layout, visual)
- Alignment accuracy metrics

## Text Comparison Thresholds

### Semantic Similarity
- **text_similarity_threshold**: 0.82 (default)
  - Cosine similarity threshold for considering text unchanged
  - Values below this indicate text modification
  - Range: 0.0-1.0

- **semantic_change_threshold**: 0.5 (default)
  - Similarity threshold below which text is considered semantically changed
  - Used for distinguishing semantic vs. minor text changes

### Text Normalization
- Case-insensitive comparison
- Unicode normalization (NFD → NFC)
- Whitespace normalization

## Formatting Comparison Thresholds

### Font Size Changes
- **font_size_change_threshold_pt**: 1.0 point (default)
  - Minimum absolute font size difference to detect as change
  - PDF requirement: changes >1pt are detected

- **formatting_change_threshold**: 0.1 (10% default)
  - Relative difference threshold for font/spacing changes
  - Used for relative comparisons

### Color Changes
- **color_difference_threshold**: 10 (default)
  - RGB color difference threshold
  - Sum of absolute differences in RGB channels
  - Range: 0-255 per channel

### Spacing Changes
- **spacing_change_threshold_ratio**: 0.1 (10% default)
  - Relative spacing change threshold
  - Detects line/paragraph spacing changes

### Page Layout Changes
- **page_size_change_threshold**: 0.05 (5% default)
  - Relative page dimension change threshold
  - Detects changes in page width/height

## Layout Comparison Thresholds

### Block Alignment
- **block_alignment_distance_threshold**: 100.0 points (default)
  - Maximum distance in points for block alignment matching
  - Used in section/block alignment algorithms

- **block_alignment_distance_limit**: 100.0 points (default)
  - Maximum distance limit for block alignment

### Page Alignment
- **page_alignment_confidence_threshold**: 0.5 (default)
  - Minimum confidence for page alignment matching
  - Range: 0.0-1.0

- **page_similarity_search_window**: 3 pages (default)
  - Search window size for page similarity matching
  - Searches ±3 pages around expected position

## Table Comparison Thresholds

### Table Matching
- **table_overlap_threshold**: 0.3 (30% default)
  - Minimum overlap score (IoU) for matching tables between pages
  - Range: 0.0-1.0

- **table_structure_confidence_threshold**: 0.5 (default)
  - Minimum confidence for table structure detection
  - Range: 0.0-1.0

## Figure Comparison Thresholds

### Figure Matching
- **figure_overlap_threshold**: 0.3 (30% default)
  - Minimum overlap score (IoU) for matching figures between pages
  - Range: 0.0-1.0

### Caption Detection
- **caption_search_margin**: 50.0 points (default)
  - Margin in points for searching figure captions near figures

- **caption_search_distance**: 100.0 points (default)
  - Maximum distance in points below figure to search for captions

## Header/Footer Detection Thresholds

### Region Detection
- **header_region_height_ratio**: 0.1 (10% default)
  - Ratio of page height for header region (top 10%)

- **footer_region_height_ratio**: 0.1 (10% default)
  - Ratio of page height for footer region (bottom 10%)

### Repetition Detection
- **header_footer_repetition_threshold**: 0.33 (33% default)
  - Minimum ratio of pages where header/footer must appear
  - To be considered a repeating header/footer

- **header_footer_match_threshold**: 0.3 (default)
  - Minimum similarity score for matching headers/footers between pages
  - Range: 0.0-1.0

## Visual Diff Thresholds

### Pixel-Level Comparison
- **visual_diff_pixel_threshold**: 30 (default)
  - Pixel difference threshold for visual diff noise filtering
  - Range: 0-255
  - Differences below this threshold are filtered out

- **visual_diff_dpi**: 150 (default)
  - DPI for visual diff rendering
  - Lower DPI = faster processing
  - Higher DPI = more accurate but slower

### Heatmap Overlay
- **heatmap_overlay_alpha**: 0.4 (40% default)
  - Transparency for heatmap overlay
  - Range: 0.0 (transparent) to 1.0 (opaque)

## UI Rendering Thresholds

### Display Settings
- **render_dpi**: 144 (default)
  - DPI for PDF page rendering in UI

- **render_scale_factor**: 2.0 (default)
  - Scale factor for high-DPI rendering
  - Improves quality on high-DPI displays

## Model Configuration

### Batch Processing
- **batch_size**: 32 (default)
  - Batch size for embedding computation
  - Larger batches = faster but more memory

- **num_workers**: 4 (default)
  - Parallel workers for OCR/alignment
  - CPU-bound parallel processing

### GPU Acceleration
- **use_gpu**: False (default)
  - Enable GPU acceleration if available
  - Requires CUDA-capable GPU and appropriate libraries

## Configuration File

All thresholds and metrics can be configured via:
- `config/settings.py` - Main configuration file
- `.env` file - Environment variable overrides
- Command-line arguments (if supported)

## Example Configuration

```python
# High-precision mode (slower but more accurate)
text_similarity_threshold = 0.90
visual_diff_dpi = 300
visual_diff_pixel_threshold = 20

# Fast mode (faster but may miss subtle changes)
text_similarity_threshold = 0.75
visual_diff_dpi = 100
visual_diff_pixel_threshold = 50
```

## Threshold Tuning Guide

1. **Too many false positives**:
   - Increase similarity thresholds
   - Increase pixel difference threshold
   - Increase overlap thresholds

2. **Missing changes**:
   - Decrease similarity thresholds
   - Decrease pixel difference threshold
   - Decrease overlap thresholds

3. **Performance issues**:
   - Decrease DPI settings
   - Increase batch size (if memory allows)
   - Enable GPU acceleration
   - Reduce search windows

## References

All thresholds are based on requirements from:
- PDF Document Comparison AI Prototype Plan (1).pdf
- Best practices from document AI research
- Empirical tuning on test datasets

