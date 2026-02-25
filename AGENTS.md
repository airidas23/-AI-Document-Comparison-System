# AGENTS.md

## Cursor Cloud specific instructions

### Product overview

AI Document Comparison System — a single-service Python (Gradio) application that compares two PDF documents and produces interactive visual diffs. No database, no Docker, no separate frontend/backend.

### Running the application

- **Start**: `source .venv/bin/activate && python app.py` (serves on port 7860, auto-fallback to 7861-7865)
- **Config**: `.env` copied from `.env.example`; defaults work out of the box
- See `README.md` for full setup and usage instructions.

### Testing

- `pytest` (runs tests from `tests/` directory per `pytest.ini`)
- `tests/test_generator_quick.py` has a pre-existing broken import (`data.synthetic.generator` module does not exist); skip with `--ignore=tests/test_generator_quick.py`
- 2 tests in `test_tesseract_bbox_overlap.py` fail due to missing test data PDFs in `data/test/lt_scan_demo/` — this is a pre-existing issue
- Integration/slow tests are skipped by default (need model downloads or env flags like `RUN_DEEPSEEK_DIRECT=1`)
- No formal linter config exists; `ruff check --select=E,F` can be used for basic checks

### Non-obvious gotchas

- **PaddleOCR requires `paddlepaddle`**: The `requirements.txt` installs `paddleocr` but not `paddlepaddle` (the compute backend). Install separately: `pip install paddlepaddle`. Without it, PaddleOCR is skipped and Tesseract is used as fallback.
- **System deps**: `tesseract-ocr` and `poppler-utils` must be installed via apt. Without them, OCR and PDF-to-image conversion fail.
- **Model downloads**: Sentence-transformer (`models/all-MiniLM-L6-v2`) and DocLayout-YOLO (`models/doclayout_yolo_docstructbench_imgsz1024.pt`) are required. Run `python scripts/setup_models.py` and `python download_doclayout_model.py`. DeepSeek-OCR is disabled by default and its ~500MB download can be skipped.
- **First PaddleOCR request is slow**: PaddleOCR warmup downloads model files to `~/.paddlex/` on first run (~2-8s). Subsequent starts are faster.
- **MCP server warning**: The `mcp_server=True` flag in `app.py` shows a non-fatal warning if `gradio[mcp]` is not installed. This is optional and does not affect core functionality.
