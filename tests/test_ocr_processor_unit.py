from __future__ import annotations

from pathlib import Path

import pytest


def test_ocr_processor_deprecated_wrapper_forwards(monkeypatch, tmp_path: Path):
    from extraction import ocr_processor

    called = {"n": 0, "arg": None}

    def fake_ocr_pdf_multi(path, *args, **kwargs):
        called["n"] += 1
        called["arg"] = path
        return ["ok"]

    monkeypatch.setattr("extraction.ocr_router.ocr_pdf_multi", fake_ocr_pdf_multi)

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%fake\n")

    with pytest.warns(DeprecationWarning):
        out = ocr_processor.ocr_pdf(pdf)

    assert out == ["ok"]
    assert called["n"] == 1
    assert called["arg"] == pdf
