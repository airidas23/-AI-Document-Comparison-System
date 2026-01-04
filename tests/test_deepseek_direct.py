"""Opt-in smoke test for DeepSeek OCR model loading + generation.

This is intentionally skipped by default because it requires:
- local DeepSeek model weights (multi-GB) or network access
- torch + transformers

Run manually with:
  RUN_DEEPSEEK_DIRECT=1 pytest -q tests/test_deepseek_direct.py
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.integration
def test_deepseek_generation_opt_in():
    if os.environ.get("RUN_DEEPSEEK_DIRECT") != "1":
        pytest.skip("Set RUN_DEEPSEEK_DIRECT=1 to run this integration test")

    torch = pytest.importorskip("torch")
    _ = torch  # keep linters quiet
    pytest.importorskip("transformers")

    from PIL import Image

    from config.settings import settings
    from extraction.deepseek_ocr_engine import DeepSeekOCR

    model_path = os.environ.get("DEEPSEEK_OCR_MODEL_PATH", settings.deepseek_ocr_model_path)
    if not model_path:
        pytest.skip("DEEPSEEK_OCR_MODEL_PATH or settings.deepseek_ocr_model_path is required")

    ocr = DeepSeekOCR(model_path)
    ocr._load_model()

    img = Image.new("RGB", (800, 1000), color="white")
    blocks = ocr.recognize(img)
    assert isinstance(blocks, list)
