"""Opt-in integration test for DocLayout-YOLO model loading.

This is intentionally skipped by default because it requires local model weights.
Run manually with:
  RUN_MODEL_DIRECT=1 pytest -q tests/test_model_direct.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.mark.integration
def test_model_direct_loading_opt_in():
    if os.environ.get("RUN_MODEL_DIRECT") != "1":
        pytest.skip("Set RUN_MODEL_DIRECT=1 to run this integration test")

    model_path = Path("models/doclayout_yolo_docstructbench_imgsz1024.pt")
    if not model_path.exists():
        pytest.skip(f"Model not found: {model_path}")

    ultralytics = pytest.importorskip("ultralytics")
    YOLO = getattr(ultralytics, "YOLO")

    model = YOLO(str(model_path))
    assert getattr(model, "names", None) is not None
