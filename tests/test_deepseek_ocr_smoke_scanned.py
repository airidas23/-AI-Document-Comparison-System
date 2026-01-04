import os
from pathlib import Path

import pytest

from config.settings import settings
from extraction.ocr_router import is_deepseek_available, ocr_pdf_multi


@pytest.mark.integration
@pytest.mark.slow
def test_deepseek_ocr_smoke_scanned_auto_fallback(monkeypatch):
    """Smoke-test DeepSeek OCR on a real scanned PDF.

    This is intentionally opt-in because it requires:
    - local DeepSeek model weights (~6GB)
    - slow inference on Apple Silicon (MPS)

    Enable with:
        RUN_DEEPSEEK_OCR=1 pytest -q tests/test_deepseek_ocr_smoke_scanned.py

    Optional tuning:
        DEEPSEEK_TEST_TIMEOUT_SEC=30

    What it asserts:
    - DeepSeek is attempted first (when available)
    - We still return usable OCR output (DeepSeek or deterministic fallback)
    - Per-page metadata contains `ocr_policy` + `ocr_attempts`
    """

    if os.environ.get("RUN_DEEPSEEK_OCR") != "1":
        pytest.skip("Set RUN_DEEPSEEK_OCR=1 to run this integration test")

    if not is_deepseek_available():
        pytest.skip("DeepSeek dependencies not available in this environment")

    # Ensure local model directory exists (recommended for Apple Silicon)
    model_dir = Path(settings.deepseek_ocr_model_path).expanduser()
    if not model_dir.exists():
        pytest.skip(
            f"DeepSeek model dir missing: {model_dir}. "
            "Run: python scripts/download_deepseek_ocr_fresh.py"
        )

    scanned_pdf = Path(
        "data/synthetic/test_scanned_dataset/variation_01/variation_01_original_scanned.pdf"
    )
    if not scanned_pdf.exists():
        pytest.skip(f"Scanned test PDF missing: {scanned_pdf}")

    timeout_sec = int(os.environ.get("DEEPSEEK_TEST_TIMEOUT_SEC", "30"))

    # Force routing mode that matches how the app should behave on Mac:
    # DeepSeek-first, but deterministic fallback if it's too slow.
    monkeypatch.setattr(settings, "ocr_scanned_policy", "auto_fallback", raising=False)
    monkeypatch.setattr(settings, "ocr_scanned_fallback_chain", ["paddle", "tesseract"], raising=False)
    monkeypatch.setattr(settings, "deepseek_timeout_sec_per_page", timeout_sec, raising=False)
    monkeypatch.setattr(settings, "deepseek_hard_timeout", True, raising=False)

    pages = ocr_pdf_multi(scanned_pdf, engine_priority=["deepseek"], prefer_digital=False)

    assert pages, "No pages returned from OCR"

    total_blocks = sum(len(p.blocks or []) for p in pages)
    total_chars = sum(len((b.text or "")) for p in pages for b in (p.blocks or []))

    assert total_blocks > 0, "OCR produced 0 blocks"
    assert total_chars > 0, "OCR produced 0 characters"

    # Metadata expectations (added by the routing work)
    for page in pages:
        md = page.metadata or {}
        assert md.get("ocr_policy") == "auto_fallback"
        assert isinstance(md.get("ocr_attempts"), list)
        assert md.get("ocr_attempts"), "Expected at least one engine attempt"

        # DeepSeek should be attempted first when available.
        first_attempt = md["ocr_attempts"][0]
        assert first_attempt.get("engine") == "deepseek"

        # Engine used can be deepseek (success) OR a fallback (timeout/insufficient).
        used = md.get("ocr_engine_used")
        assert used in {"deepseek", "paddle", "tesseract"}

        if used != "deepseek":
            # If we fell back, ensure the attempts include a DeepSeek failure/insufficient outcome.
            deepseek_attempts = [a for a in md["ocr_attempts"] if a.get("engine") == "deepseek"]
            assert deepseek_attempts
            assert deepseek_attempts[0].get("outcome") in {"failed", "insufficient"}
