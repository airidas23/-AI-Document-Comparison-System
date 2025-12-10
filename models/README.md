# Models Directory

This directory contains locally downloaded model weights for offline use.

## Required Models

### DeepSeek-OCR
- **Location**: `models/deepseek-ocr/`
- **Source**: HuggingFace `deepseek-ai/deepseek-ocr`
- **Size**: ~500MB
- **Purpose**: OCR processing for scanned PDF documents
- **Download**: Run `python scripts/setup_models.py` or download manually from HuggingFace

## Layout Detection Models

### DocLayout-YOLO (Document Layout Analysis) ⭐ PRIMARY MODEL

**Current Model**: `doclayout_yolo_docstructbench_imgsz1024.pt` (38.82 MB)

**Purpose**: Document-specific layout detection optimized for PDF analysis

**Source**: [juliozhao/DocLayout-YOLO-DocStructBench](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench)

**Detected Classes (10)**:
- `title` - Document titles
- `plain text` - Regular text paragraphs
- `abandon` - Content to ignore
- `figure` - Images and diagrams  
- `figure_caption` - Figure captions
- `table` - Tables
- `table_caption` - Table captions
- `table_footnote` - Table footnotes
- `isolate_formula` - Mathematical formulas
- `formula_caption` - Formula captions

**Performance**:
- Inference speed: ~120-160ms per page
- Trained on DocSynth-300K dataset
- Optimized for document layout analysis

**Installation**:
```bash
pip install doclayout-yolo
python download_doclayout_model.py
```

**Configuration** (`.env`):
```
YOLO_LAYOUT_MODEL_NAME=models/doclayout_yolo_docstructbench_imgsz1024.pt
YOLO_LAYOUT_CONFIDENCE=0.3
```



### Sentence Transformer (all-MiniLM-L6-v2)
- **Location**: `models/all-MiniLM-L6-v2/`
- **Source**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Size**: ~80MB
- **Purpose**: Semantic text comparison and embedding generation
- **Download**: Run `python scripts/setup_models.py` or download manually from HuggingFace

## Download Methods

### Automated Download (Recommended)

Run the setup script from the project root:

```bash
python scripts/setup_models.py
```

This will download both models to their respective directories. The script is idempotent - safe to run multiple times.

### Manual Download

#### Using HuggingFace Hub CLI

1. Install HuggingFace Hub:
   ```bash
   pip install huggingface_hub
   ```

2. Download DeepSeek-OCR:
   ```bash
   huggingface-cli download deepseek-ai/deepseek-ocr --local-dir models/deepseek-ocr
   ```

3. Download Sentence Transformer:
   ```bash
   huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir models/all-MiniLM-L6-v2
   ```

#### Using Python

```python
from huggingface_hub import snapshot_download

# Download DeepSeek-OCR
snapshot_download(
    repo_id="deepseek-ai/deepseek-ocr",
    local_dir="models/deepseek-ocr",
)

# Download Sentence Transformer
snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir="models/all-MiniLM-L6-v2",
)
```

## Directory Structure

After downloading, your `models/` directory should look like:

```
models/
├── .gitkeep
├── README.md
├── deepseek-ocr/
│   ├── config.json
│   ├── pytorch_model.bin (or model.safetensors)
│   └── ... (other model files)
└── all-MiniLM-L6-v2/
    ├── config.json
    ├── pytorch_model.bin (or model.safetensors)
    └── ... (other model files)
```

## Configuration

After downloading models, ensure your `.env` file points to the local paths:

```env
DEEPSEEK_OCR_MODEL_PATH=models/deepseek-ocr
SENTENCE_TRANSFORMER_MODEL=models/all-MiniLM-L6-v2
```

If you prefer to use HuggingFace models directly (downloaded on first use), you can set:

```env
DEEPSEEK_OCR_MODEL_PATH=deepseek-ai/deepseek-ocr
SENTENCE_TRANSFORMER_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Size Estimates

- **DeepSeek-OCR**: ~500MB
- **Sentence Transformer**: ~80MB
- **Total**: ~580MB

Ensure you have sufficient disk space before downloading.

## Notes

- Model files are excluded from git (see `.gitignore`)
- Models are cached locally for faster subsequent loads
- First download may take time depending on internet speed
- Models can be shared between projects by using the same directory structure

