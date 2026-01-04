import fitz
import pytest
from unittest.mock import patch

from config.settings import settings


def _make_one_page_pdf(tmp_path) -> str:
    pdf_path = tmp_path / "one_page.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), "test")
    doc.save(pdf_path)
    doc.close()
    return str(pdf_path)


@pytest.fixture
def restore_settings():
    snapshot = {
        "ocr_scanned_policy": getattr(settings, "ocr_scanned_policy", "strict"),
        "ocr_scanned_fallback_chain": getattr(settings, "ocr_scanned_fallback_chain", ["tesseract", "paddle"]),
        "ocr_scanned_default_chain": getattr(settings, "ocr_scanned_default_chain", ["tesseract", "paddle", "deepseek"]),
        "ocr_fallback_enabled": getattr(settings, "ocr_fallback_enabled", True),
        "ocr_engine": getattr(settings, "ocr_engine", "paddle"),
    }
    yield
    for k, v in snapshot.items():
        setattr(settings, k, v)


def _dummy_success_pages(engine: str):
    from comparison.models import PageData, TextBlock

    long_text = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
    )
    blocks = [
        TextBlock(text=long_text[:40], bbox={"x": 10, "y": 10, "width": 500, "height": 12}, metadata={"confidence": 0.9}),
        TextBlock(text=long_text[40:80], bbox={"x": 10, "y": 30, "width": 500, "height": 12}, metadata={"confidence": 0.9}),
        TextBlock(text=long_text[80:], bbox={"x": 10, "y": 50, "width": 500, "height": 12}, metadata={"confidence": 0.9}),
    ]
    page = PageData(page_num=1, width=595, height=842, blocks=blocks)
    page.metadata = {
        "extraction_method": f"ocr_{engine}",
        "ocr_engine_used": engine,
    }
    return [page]


def test_strict_mode_no_fallback_on_timeout(tmp_path, restore_settings):
    from extraction.ocr_router import ocr_pdf_multi

    settings.ocr_scanned_policy = "strict"
    settings.ocr_fallback_enabled = True  # policy should still prevent fallback

    pdf_path = _make_one_page_pdf(tmp_path)

    seen_priority = {}

    def fake_select(priority, hardware_available=None):
        seen_priority["value"] = list(priority)
        return list(priority), {}

    with patch("extraction.ocr_router.select_ocr_engine", side_effect=fake_select) as sel, patch(
        "extraction.ocr_router._run_ocr_engine", side_effect=TimeoutError("timeout")
    ) as run:
        pages = ocr_pdf_multi(pdf_path, engine_priority=["deepseek", "tesseract"], prefer_digital=False)

    assert sel.call_count == 1
    assert seen_priority["value"] == ["deepseek"]  # strict trims
    assert run.call_count == 1
    assert run.call_args[0][1] == "deepseek"

    assert len(pages) == 1
    md = pages[0].metadata
    assert md.get("ocr_policy") == "strict"
    assert md.get("ocr_engine_selected") == "deepseek"
    assert md.get("ocr_status") == "failed"
    assert md.get("ocr_failure_reason") == "engine_timeout"
    assert isinstance(md.get("ocr_attempts"), list)
    assert md["ocr_attempts"][0]["engine"] == "deepseek"


def test_auto_fallback_after_deepseek_timeout(tmp_path, restore_settings):
    from extraction.ocr_router import ocr_pdf_multi

    settings.ocr_scanned_policy = "auto_fallback"
    settings.ocr_scanned_fallback_chain = ["tesseract", "paddle"]
    settings.ocr_fallback_enabled = True

    pdf_path = _make_one_page_pdf(tmp_path)

    def fake_run(_path, engine_name):
        if engine_name == "deepseek":
            raise TimeoutError("timeout")
        return _dummy_success_pages(engine_name)

    with patch("extraction.ocr_router.select_ocr_engine", side_effect=lambda p, hardware_available=None: (list(p), {})):
        with patch("extraction.ocr_router._run_ocr_engine", side_effect=fake_run):
            pages = ocr_pdf_multi(pdf_path, engine_priority=["deepseek", "tesseract"], prefer_digital=False)

    assert len(pages) == 1
    md = pages[0].metadata
    assert md.get("ocr_policy") == "auto_fallback"
    assert md.get("ocr_engine_selected") == "deepseek"
    assert md.get("ocr_engine_used") == "tesseract"
    assert md.get("ocr_status") == "ok"

    attempts = md.get("ocr_attempts")
    assert isinstance(attempts, list)
    assert [a["engine"] for a in attempts] == ["deepseek", "tesseract"]
    assert attempts[0]["outcome"] == "failed"
    assert attempts[0]["reason"] == "engine_timeout"
    assert attempts[1]["outcome"] == "success"


def test_engine_priority_contains_fallback_chain_in_auto(tmp_path, restore_settings):
    from extraction.ocr_router import ocr_pdf_multi

    settings.ocr_scanned_policy = "auto_fallback"
    settings.ocr_scanned_fallback_chain = ["tesseract", "paddle"]
    settings.ocr_fallback_enabled = True

    pdf_path = _make_one_page_pdf(tmp_path)

    captured = {}

    def fake_select(priority, hardware_available=None):
        captured["priority"] = list(priority)
        return list(priority), {}

    with patch("extraction.ocr_router.select_ocr_engine", side_effect=fake_select), patch(
        "extraction.ocr_router._run_ocr_engine", side_effect=RuntimeError("boom")
    ):
        _ = ocr_pdf_multi(pdf_path, engine_priority=["deepseek"], prefer_digital=False)

    assert captured["priority"][:3] == ["deepseek", "tesseract", "paddle"]


def test_auto_fallback_uses_scanned_default_chain_when_priority_not_provided(tmp_path, restore_settings):
    """Regression test: auto_fallback should respect ocr_scanned_default_chain order.

    Previously, settings.ocr_engine could be moved to the front of the chain even
    in auto_fallback, unintentionally making Paddle run before Tesseract on scanned
    PDFs and degrading diff quality.
    """
    from extraction.ocr_router import ocr_pdf_multi

    settings.ocr_scanned_policy = "auto_fallback"
    settings.ocr_fallback_enabled = True
    settings.ocr_engine = "paddle"  # should NOT reorder the scanned default chain
    settings.ocr_scanned_default_chain = ["tesseract", "paddle", "deepseek"]

    pdf_path = _make_one_page_pdf(tmp_path)

    captured = {}

    def fake_select(priority, hardware_available=None):
        captured["priority"] = list(priority)
        return list(priority), {}

    with patch("extraction.ocr_router.select_ocr_engine", side_effect=fake_select), patch(
        "extraction.ocr_router._run_ocr_engine", side_effect=RuntimeError("boom")
    ):
        _ = ocr_pdf_multi(pdf_path, engine_priority=None, prefer_digital=False)

    assert captured["priority"][:3] == ["tesseract", "paddle", "deepseek"]


def test_manifest_records_policy_and_reason(tmp_path, restore_settings):
    from extraction.ocr_router import ocr_pdf_multi
    from pipeline.compare_pdfs import ExtractionManifest

    settings.ocr_scanned_policy = "strict"
    settings.ocr_fallback_enabled = True

    pdf_path = _make_one_page_pdf(tmp_path)

    with patch("extraction.ocr_router.select_ocr_engine", side_effect=lambda p, hardware_available=None: (list(p), {})):
        with patch("extraction.ocr_router._run_ocr_engine", side_effect=TimeoutError("timeout")):
            pages = ocr_pdf_multi(pdf_path, engine_priority=["deepseek"], prefer_digital=False)

    manifest = ExtractionManifest.from_pages(pages, input_path=pdf_path, engine_used="deepseek")
    assert manifest.page_manifests
    pm = manifest.page_manifests[0]
    assert pm.get("ocr_policy") == "strict"
    assert pm.get("ocr_engine_selected") == "deepseek"
    assert pm.get("ocr_status") == "failed"
    assert pm.get("ocr_failure_reason") == "engine_timeout"
    assert isinstance(pm.get("ocr_attempts"), list)
