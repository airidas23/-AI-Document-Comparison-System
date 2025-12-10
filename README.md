# AI Document Comparison System

Local-first pipeline to compare two PDF documents (scanned or digital) and produce interactive diffs with content and formatting changes.

## Architecture
- Extraction: PyMuPDF for digital PDFs, multi-engine OCR routing (DeepSeek-OCR for CUDA environments, PaddleOCR/Tesseract for Mac/CPU) for scanned pages, optional layout analysis.
- Alignment & Comparison: Embedding-based text diff, formatting/layout checks with normalized style comparison, pixel-level visual diff.
- Visualization: Gradio UI with side-by-side viewer, diff navigator, and heatmap overlays.

The OCR system automatically selects the best engine based on hardware availability: DeepSeek-OCR requires CUDA and is used on GPU-enabled systems, while PaddleOCR (primary) and Tesseract (fallback) work on Mac and CPU-only environments.

## Project Structure
See `project/` tree in the request; key modules are scaffolded under `extraction/`, `comparison/`, `visualization/`, and `export/`.


## Setup

1. **Environment Setup**
    Create a virtual environment and install dependencies:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
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


    **Step B: Download Layout Analysis Model (Primary)**
    ```bash
    python download_doclayout_model.py
    ```
    This downloads:
    - DocLayout-YOLO (~39MB) to `models/` - *Required for document layout analysis*

    > **Note**: Total download size is ~700MB. Models are saved in the `models/` directory.

3. **Configuration**
    Copy the example configuration file:
    ```bash
    cp .env.example .env
    ```
    The default settings in `.env` are pre-configured to use the local models you just downloaded.

## Running the System

1. **Start the Application**
    ```bash
    python app.py
    ```

2. **Access the UI**
    Open your browser and navigate to:
    - **Local**: [http://localhost:7860](http://localhost:7860)

3. **Usage Tips**
    - **Digital PDFs**: Drag & drop files and click "Compare".
    - **Scanned PDFs**: Enable "Scanned Document Mode" checkbox to activate OCR.
    - **Detailed View**: Use the "Synchronized Viewer" for side-by-side inspection.

## Performance Notes
- **Ocr Warmup**: The first run might be slightly slower as OCR engines initialize.
- **GPU Acceleration**: DeepSeek-OCR automatically uses CUDA if available. On Mac M-series, it runs on CPU/MPS (via custom implementation).
- **Processing Time**: Target is <3s per page. Scanned documents take longer due to OCR.

## Testing
Run the test suite to verify the installation:
```bash
pytest
```
