# Models and Tools Comparison

This document describes the models and tools used in the PDF Document Comparison System, their pros/cons, and implementation status.

## Text Extraction Models

### PyMuPDF (fitz)
**Status**: ✅ Implemented

**Pros**:
- Fast extraction from digital PDFs
- Preserves text positions and formatting information
- Handles most PDF formats well
- Lightweight, no external dependencies

**Cons**:
- Cannot extract text from scanned PDFs
- Limited layout analysis capabilities

**Usage**: Primary extraction method for digital PDFs

### DeepSeek-OCR
**Status**: ✅ Implemented

**Pros**:
- Handles scanned documents
- Local processing (privacy-preserving)
- Good accuracy for printed text

**Cons**:
- Slower than native text extraction
- Requires model weights (~500MB)
- May struggle with handwritten text or poor quality scans

**Usage**: Fallback for scanned PDFs when no extractable text is found

### Alternative: Tesseract OCR
**Status**: ⚠️ Not implemented (DeepSeek-OCR preferred)

**Pros**:
- Open source, widely available
- Good for many languages
- Can be fine-tuned

**Cons**:
- Generally lower accuracy than modern deep learning OCR
- Requires language packs
- Slower processing

## Text Comparison Models

### Sentence Transformers (all-MiniLM-L6-v2)
**Status**: ✅ Implemented

**Pros**:
- Semantic understanding (catches paraphrasing)
- Fast inference
- Local processing
- Good balance between speed and accuracy
- Small model size (~80MB)

**Cons**:
- May miss subtle meaning changes
- Requires GPU for best performance on large batches

**Alternatives Considered**:
- **all-mpnet-base-v2**: Better accuracy but slower
- **all-MiniLM-L12-v2**: Better accuracy but larger

**Usage**: Default for semantic text comparison

## Layout Analysis Models

### LayoutParser
**Status**: ⚠️ Partially Implemented (framework ready, model loading needed)

**Pros**:
- Pre-trained models available (PubLayNet, DocLayNet)
- Good accuracy for structured documents
- Detects tables, figures, headings, paragraphs

**Cons**:
- Requires model download (~500MB+)
- Slower than heuristic methods
- May not work well on all document types

**Models Available**:
- `lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config`: Academic papers
- `lp://DocLayNet/faster_rcnn_R_50_FPN_3x/config`: General documents

**Usage**: Optional enhancement for better layout detection

### Heuristic Layout Detection (Current Implementation)
**Status**: ✅ Implemented

**Pros**:
- Fast, no model loading
- Works immediately
- Good for simple layouts

**Cons**:
- Lower accuracy than ML models
- May miss complex table structures
- Limited understanding of document semantics

**Usage**: Fallback when LayoutParser unavailable

## Table Detection Models

### Table Transformer
**Status**: ⚠️ Not Implemented (placeholder for future)

**Pros**:
- State-of-the-art table structure detection
- Handles complex table layouts
- Detects merged cells, nested tables

**Cons**:
- Requires PyTorch and model weights
- Larger model size
- Slower inference

**GitHub**: https://github.com/microsoft/table-transformer

**Usage**: Future enhancement for better table comparison

### PyMuPDF/Camelot Table Detection
**Status**: ✅ Partially Implemented (basic heuristic)

**Pros**:
- Integrated with PDF parsing
- Fast detection
- Good for simple tables

**Cons**:
- Limited to grid-like tables
- Struggles with complex layouts
- Lower accuracy

**Usage**: Current implementation for table detection

## Visual Comparison

### OpenCV-based Pixel Diff
**Status**: ✅ Implemented

**Pros**:
- Catches visual differences missed by text comparison
- Handles formatting changes
- Fast with proper thresholding

**Cons**:
- Pixel-level differences may not be semantically meaningful
- Requires threshold tuning
- May generate noise

**Usage**: Visual diff heatmap generation

## Alignment Algorithms

### Jaccard Similarity (Word Overlap)
**Status**: ✅ Implemented

**Pros**:
- Fast computation
- Good for page alignment
- Simple and interpretable

**Cons**:
- May not catch semantic similarity
- Sensitive to word choice differences

**Usage**: Page and section alignment

### Semantic Embedding Alignment
**Status**: ✅ Implemented (via sentence transformers)

**Pros**:
- Understands semantic similarity
- Better for rephrased content

**Cons**:
- Slower than word overlap
- Requires model loading

**Usage**: Enhanced page alignment when page counts differ

## Model Configuration

All models can be configured via `config/settings.py`:

```python
# Text comparison
sentence_transformer_model = "sentence-transformers/all-MiniLM-L6-v2"

# Layout analysis
layoutparser_model = None  # Set to model name to enable

# Table detection
table_transformer_model_path = None  # Set to path to enable

# OCR
deepseek_ocr_model_path = "models/deepseek-ocr"
```

## Performance Considerations

- **Text extraction**: PyMuPDF is fastest for digital PDFs
- **OCR**: DeepSeek-OCR is faster than Tesseract but requires GPU for best performance
- **Embeddings**: Sentence transformers benefit from GPU acceleration
- **Layout analysis**: Heuristic methods are faster but less accurate than ML models
- **Visual diff**: OpenCV operations are CPU-bound but fast with proper thresholding

## Future Enhancements

1. **Table Transformer integration** for better table structure detection
2. **LayoutParser model loading** for improved layout analysis
3. **Alternative sentence transformer models** for better accuracy
4. **GPU acceleration** for all ML models
5. **Model caching** to avoid reloading between comparisons

