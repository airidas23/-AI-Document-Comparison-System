# AI Document Comparison System

Local-first pipeline to compare two PDF documents (scanned or digital) and produce interactive diffs with content and formatting changes.

## Key Features
- **Multi-Engine OCR**: Intelligent routing between DeepSeek-OCR (GPU/MPS), PaddleOCR (Primary CPU), and Tesseract (Fallback).
- **Layout Analysis**: Advanced document structure detection using DocLayout-YOLO.
- **Smart Comparison**: Embedding-based semantic text diff, formatting/layout checks, and table-aware comparison.
- **Visual Diff**: Pixel-level heatmap overlays for quick visual inspection of changes.
- **Interactive UI**: Gradio-based interface with side-by-side synchronized viewer and diff navigator.

## Project Structure
- `extraction/`: OCR engines, layout analysis, and PDF parsing logic.
- `comparison/`: Text alignment, formatting comparison, and diff classification.
- `visualization/`: Gradio UI components and heatmap generation.
- `export/`: JSON and PDF export functionality.
- `models/`: Local storage for AI model weights.
- `scripts/`: Setup, evaluation, and utility scripts.

## Setup

1. **Environment Setup**
    Create a virtual environment and install dependencies:
    ```bash
    # macOS / Linux
    python3 -m venv .venv
    source .venv/bin/activate
    
    # Windows
    python -m venv .venv
    .venv\Scripts\activate
    
    # Install dependencies
    pip install -r requirements.txt
    ```

    *(Optional) For MCP (Model Context Protocol) support:*
    ```bash
    pip install -r requirements-mcp.txt
    ```

2. **Model Download**
    The system requires several AI models to run locally. Download them using the provided scripts:

    **Step A: Download Base Models (OCR & Embeddings)**
    ```bash
    python scripts/setup_models.py
    ```
    This downloads:
    - DeepSeek-OCR (~500MB)
    - Sentence Transformer (~80MB)
    - SAM ViT-H (Optional, for advanced segmentation)

    **Step B: Download Layout Analysis Model**
    ```bash
    python download_doclayout_model.py
    ```
    This downloads:
    - DocLayout-YOLO (~39MB) - *Required for document layout analysis*

    > **Note**: Total download size is ~700MB. Models are saved in the `models/` directory.

3. **Configuration**
    Copy the example configuration file:
    ```bash
    cp .env.example .env
    ```
    The default settings in `.env` are pre-configured to use the local models. You can adjust `OCR_ENGINE` (options: `paddle`, `tesseract`, `deepseek`) based on your hardware.

## Running the System

1. **Start the Application**
    ```bash
    # macOS / Linux
    python app.py
    # OR
    ./run_app.sh

    # Windows
    python app.py
    ```

2. **Access the UI**
    Open your browser and navigate to:
    - **Local**: [http://localhost:7860](http://localhost:7860) or http://localhost:7861 (if port 7860 is in use).

3. **Usage Tips**
    - **Digital PDFs**: Drag & drop files and click "Compare".
    - **Scanned PDFs**: Enable "Scanned Document Mode" checkbox to activate OCR.
    - **Synchronized Viewer**: Use this for side-by-side inspection of changes.

## Documentation
For more detailed information, see the following documents:
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md): Detailed architecture and roadmap.
- [Models Guide](docs/MODELS.md): Information about the AI models used in the system.
- [Metrics & Thresholds](docs/METRICS_AND_THRESHOLDS.md): Explanation of comparison metrics and how to tune them.
- [OCR Performance Analysis](docs/OCR_PERFORMANCE_ANALYSIS.md): Benchmarks and analysis of different OCR engines.
- [Gradio MCP Integration](docs/GRADIO_MCP.md): Guide for using the system via Model Context Protocol.
- [Status Review](docs/STATUS_REVIEW.md): Current project status and next steps.

## Troubleshooting

### Apple Silicon (M1-M4) Issues
If you encounter `dtype mismatch` or `MPS` errors with DeepSeek-OCR:
1. Ensure you have the latest model weights (see [DeepSeek MPS Fix](docs/DEEPSEEK_MPS_FIX.md)).
2. Run the verification script:
   ```bash
   python scripts/verify_deepseek_model.py
   ```
3. If problems persist, switch to `paddle` or `tesseract` in your `.env` file.

### Missing Dependencies
If an OCR engine is skipped, check if the required system libraries are installed:
- **Tesseract**:
    - **macOS**: `brew install tesseract`
    - **Linux**: `sudo apt install tesseract-ocr`
    - **Windows**: Download and run the installer from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). Add the installation path (usually `C:\Program Files\Tesseract-OCR`) to your System PATH.
- **Poppler** (Required for PDF rendering):
    - **macOS**: `brew install poppler`
    - **Linux**: `sudo apt install poppler-utils`
    - **Windows**: Download from [Release page](https://github.com/oschwartz10612/poppler-windows/releases), extract, and add the `bin` folder to your System PATH.
- **PaddleOCR**: Requires `paddlepaddle`. If installation fails on Mac, the system will automatically fall back to Tesseract.

## Performance Notes
- **OCR Warmup**: The first run might be slightly slower as OCR engines initialize in the background.
- **Hardware Acceleration**: DeepSeek-OCR automatically uses CUDA if available. On Mac M-series, it runs on CPU/MPS.
- **Processing Time**: Target is <3s per page for digital PDFs. Scanned documents take longer due to OCR processing.

## Testing
Run the test suite to verify the installation:
```bash
pytest
```
