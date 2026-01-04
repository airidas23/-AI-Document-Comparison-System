import sys
import types
from pathlib import Path

import pytest

from comparison.models import PageData, TextBlock


def _mk_page(page_num: int = 1, *, blocks=None, width: float = 600, height: float = 800) -> PageData:
    return PageData(page_num=page_num, width=width, height=height, blocks=list(blocks or []), metadata={})


def _mk_block(text: str, *, conf=None) -> TextBlock:
    md = {}
    if conf is not None:
        md["confidence"] = conf
    return TextBlock(text=text, bbox={"x": 10, "y": 10, "width": 100, "height": 20}, metadata=md)


def test_apply_scanned_policy_strict_returns_single_engine(monkeypatch):
    from extraction import ocr_router

    monkeypatch.setattr(ocr_router.settings, "ocr_engine", "tesseract", raising=False)

    priority, allow = ocr_router._apply_scanned_policy_to_priority(
        ["deepseek", "paddle", "tesseract"], policy="strict"
    )
    assert priority == ["deepseek"]
    assert allow is False


def test_apply_scanned_policy_auto_fallback_expands_deepseek_chain(monkeypatch):
    from extraction import ocr_router

    monkeypatch.setattr(ocr_router.settings, "ocr_scanned_fallback_chain", ["tesseract", "paddle"], raising=False)

    priority, allow = ocr_router._apply_scanned_policy_to_priority(["deepseek"], policy="auto_fallback")
    assert priority[:3] == ["deepseek", "tesseract", "paddle"]
    assert allow is True


def test_classify_failure_reason_stable_tokens():
    from extraction import ocr_router

    assert ocr_router._classify_failure_reason(TimeoutError("x")) == "engine_timeout"
    assert ocr_router._classify_failure_reason("insufficient blocks") == "insufficient_blocks"
    assert ocr_router._classify_failure_reason("OOM") == "engine_memory"
    # Note: substring match treats "boom" as OOM-like.
    assert ocr_router._classify_failure_reason(RuntimeError("boom")) == "engine_memory"


def test_select_ocr_engine_skips_deepseek_without_acceleration(monkeypatch):
    from extraction import ocr_router

    monkeypatch.setattr(ocr_router, "is_mps_available", lambda: False)
    monkeypatch.setattr(ocr_router, "is_deepseek_available", lambda: True)
    monkeypatch.setattr(ocr_router, "is_tesseract_available", lambda: True)
    monkeypatch.setattr(ocr_router, "is_paddle_available", lambda: True)

    engines, skipped = ocr_router.select_ocr_engine(
        ["deepseek", "tesseract", "paddle"], hardware_available={"cuda": False, "mps": False}
    )
    assert engines == ["tesseract", "paddle"]
    assert skipped.get("deepseek") == "cuda_or_mps_unavailable"


def test_select_ocr_engine_skips_deepseek_when_dependency_missing(monkeypatch):
    from extraction import ocr_router

    monkeypatch.setattr(ocr_router, "is_deepseek_available", lambda: False)
    monkeypatch.setattr(ocr_router, "is_tesseract_available", lambda: True)

    engines, skipped = ocr_router.select_ocr_engine(
        ["deepseek", "tesseract"], hardware_available={"cuda": True, "mps": False}
    )
    assert engines == ["tesseract"]
    assert skipped.get("deepseek") == "dependency_missing"


def test_annotate_ocr_quality_flags_low_confidence(monkeypatch):
    from extraction import ocr_router

    monkeypatch.setattr(ocr_router.settings, "ocr_quality_min_chars_per_page", 25, raising=False)
    monkeypatch.setattr(ocr_router.settings, "ocr_quality_min_avg_confidence", 0.55, raising=False)
    monkeypatch.setattr(ocr_router.settings, "ocr_quality_max_gibberish_ratio", 0.35, raising=False)

    pages = [_mk_page(blocks=[_mk_block("short", conf=0.1)])]
    ocr_router._annotate_ocr_quality(pages, engine_name="tesseract")

    md = pages[0].metadata["ocr_quality"]
    assert md["engine"] == "tesseract"
    assert md["low_confidence"] is True
    assert pages[0].metadata["ocr_low_confidence"] is True


def test_is_ocr_successful_respects_low_confidence_rejection(monkeypatch):
    from extraction import ocr_router

    monkeypatch.setattr(ocr_router.settings, "min_ocr_blocks_per_page", 1, raising=False)

    p1 = _mk_page(blocks=[_mk_block("hello world")])
    p1.metadata["ocr_quality"] = {"low_confidence": True}

    p2 = _mk_page(page_num=2, blocks=[_mk_block("more text")])
    p2.metadata["ocr_quality"] = {"low_confidence": True}

    assert ocr_router._is_ocr_successful([p1, p2], engine_name="tesseract") is False


def test_is_ocr_successful_accepts_when_any_page_not_low_conf(monkeypatch):
    from extraction import ocr_router

    monkeypatch.setattr(ocr_router.settings, "min_ocr_blocks_per_page", 1, raising=False)

    p1 = _mk_page(blocks=[_mk_block("hello")])
    p1.metadata["ocr_quality"] = {"low_confidence": True}

    p2 = _mk_page(page_num=2, blocks=[_mk_block("ok")])
    p2.metadata["ocr_quality"] = {"low_confidence": False}

    assert ocr_router._is_ocr_successful([p1, p2], engine_name="tesseract") is True


def test_annotate_routing_metadata_sets_expected_fields():
    from extraction import ocr_router

    pages = [_mk_page(blocks=[_mk_block("x")])]
    ocr_router._annotate_routing_metadata(
        pages,
        policy="strict",
        engine_selected="tesseract",
        attempts=[{"engine": "tesseract", "outcome": "success"}],
        engine_priority=["tesseract"],
        available_engines=["tesseract"],
        preflight_skipped={"deepseek": "cuda_or_mps_unavailable"},
        status="ok",
        failure_reason=None,
    )

    md = pages[0].metadata
    assert md["ocr_policy"] == "strict"
    assert md["ocr_engine_selected"] == "tesseract"
    assert md["ocr_status"] == "ok"
    assert md["ocr_attempts"][0]["outcome"] == "success"


def test_add_layout_metadata_success_with_fake_analyzer(monkeypatch):
    from extraction import ocr_router

    def _analyze_layout(_path: Path):
        p1 = _mk_page(1, blocks=[_mk_block("a")])
        p1.metadata = {
            "tables": ["t"],
            "figures": ["f"],
            "text_blocks": ["tb"],
            "layout_method": "yolo",
        }
        p2 = _mk_page(2, blocks=[_mk_block("b")])
        p2.metadata = {"tables": [], "figures": [], "text_blocks": [], "layout_method": "yolo"}
        return [p1, p2]

    fake_mod = types.SimpleNamespace(analyze_layout=_analyze_layout)

    # Inject module to avoid importing heavy dependencies.
    monkeypatch.setitem(sys.modules, "extraction.layout_analyzer", fake_mod)

    pages = [_mk_page(1, blocks=[_mk_block("x")]), _mk_page(2, blocks=[_mk_block("y")])]
    out = ocr_router._add_layout_metadata(Path("/tmp/x.pdf"), pages)

    assert out[0].metadata.get("layout_analyzed") is True
    assert out[0].metadata.get("tables") == ["t"]
    assert out[0].metadata.get("figures") == ["f"]


def test_add_layout_metadata_failure_marks_pages(monkeypatch):
    from extraction import ocr_router

    def _boom(_path):
        raise RuntimeError("no model")

    fake_mod = types.SimpleNamespace(analyze_layout=_boom)
    monkeypatch.setitem(sys.modules, "extraction.layout_analyzer", fake_mod)

    pages = [_mk_page(1, blocks=[_mk_block("x")])]
    out = ocr_router._add_layout_metadata(Path("/tmp/x.pdf"), pages)

    assert out[0].metadata.get("layout_analyzed") is False
    assert "layout_error" in out[0].metadata
    assert out[0].metadata.get("tables") == []


def test_create_empty_pages_with_metadata_without_real_pymupdf(monkeypatch, tmp_path):
    from extraction import ocr_router

    class _Rect:
        def __init__(self, w, h):
            self.width = w
            self.height = h

    class _FakePage:
        def __init__(self, number, w=600, h=800):
            self.number = number
            self.rect = _Rect(w, h)

    class _FakeDoc(list):
        def __init__(self):
            super().__init__([_FakePage(0), _FakePage(1)])

        def close(self):
            return None

    class _FakeFitz(types.SimpleNamespace):
        def open(self, _path):
            return _FakeDoc()

    monkeypatch.setitem(sys.modules, "fitz", _FakeFitz())

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    pages = ocr_router._create_empty_pages_with_metadata(
        pdf,
        ["tesseract"],
        "boom",
        policy="strict",
        engine_selected="tesseract",
        attempts=[{"engine": "tesseract", "outcome": "failed"}],
        engine_priority=["tesseract"],
        available_engines=["tesseract"],
        preflight_skipped={},
        failure_reason="engine_error",
    )

    assert len(pages) == 2
    assert pages[0].metadata["ocr_status"] == "failed"
    assert pages[0].metadata["ocr_engine_used"] == "tesseract"


def test_ocr_pdf_multi_digital_fast_path(monkeypatch, tmp_path):
    from extraction import ocr_router

    pdf = tmp_path / "d.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    sentinel = [_mk_page(1, blocks=[_mk_block("digital")])]
    monkeypatch.setattr(ocr_router, "_try_digital_extract", lambda *_a, **_k: sentinel)

    def _should_not_run(*_a, **_k):
        raise AssertionError("_run_ocr_engine should not be called")

    monkeypatch.setattr(ocr_router, "_run_ocr_engine", _should_not_run)

    out = ocr_router.ocr_pdf_multi(pdf, prefer_digital=True)
    assert out is sentinel


def test_ocr_pdf_multi_fallback_second_engine_success(monkeypatch, tmp_path):
    from extraction import ocr_router

    pdf = tmp_path / "s.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    monkeypatch.setattr(ocr_router.settings, "ocr_scanned_policy", "auto_fallback", raising=False)
    monkeypatch.setattr(ocr_router.settings, "ocr_fallback_enabled", True, raising=False)
    monkeypatch.setattr(ocr_router.settings, "min_ocr_blocks_per_page", 1, raising=False)

    monkeypatch.setattr(ocr_router, "select_ocr_engine", lambda prio: (["tesseract", "paddle"], {}))
    monkeypatch.setattr(ocr_router, "normalize_page_bboxes", lambda pages: pages)

    def _run(path: Path, engine_name: str):
        if engine_name == "tesseract":
            # Insufficient
            return [_mk_page(1, blocks=[])]
        return [_mk_page(1, blocks=[_mk_block("a" * 60, conf=0.9)])]

    monkeypatch.setattr(ocr_router, "_run_ocr_engine", _run)

    out = ocr_router.ocr_pdf_multi(pdf, prefer_digital=False, run_layout_analysis=False)
    assert out[0].metadata.get("ocr_status") == "ok"
    assert out[0].metadata.get("ocr_fallback_reason") is not None
    assert out[0].metadata.get("ocr_attempts")


def test_ocr_pdf_multi_strict_insufficient_returns_failure_pages(monkeypatch, tmp_path):
    from extraction import ocr_router

    pdf = tmp_path / "f.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    monkeypatch.setattr(ocr_router.settings, "ocr_scanned_policy", "strict", raising=False)
    monkeypatch.setattr(ocr_router.settings, "ocr_fallback_enabled", True, raising=False)
    monkeypatch.setattr(ocr_router.settings, "min_ocr_blocks_per_page", 2, raising=False)

    monkeypatch.setattr(ocr_router, "select_ocr_engine", lambda prio: (["tesseract"], {}))
    monkeypatch.setattr(ocr_router, "normalize_page_bboxes", lambda pages: pages)

    monkeypatch.setattr(ocr_router, "_run_ocr_engine", lambda *_a, **_k: [_mk_page(1, blocks=[])])

    sentinel = [_mk_page(1, blocks=[])]

    def _fake_create(*_a, **_k):
        sentinel[0].metadata["ocr_status"] = "failed"
        return sentinel

    monkeypatch.setattr(ocr_router, "_create_empty_pages_with_metadata", _fake_create)

    out = ocr_router.ocr_pdf_multi(pdf, prefer_digital=False, run_layout_analysis=False)
    assert out[0].metadata.get("ocr_status") == "failed"


def test_get_engine_render_dpi_uses_per_engine_settings(monkeypatch):
    from extraction import ocr_router

    monkeypatch.setattr(ocr_router.settings, "deepseek_render_dpi", 61, raising=False)
    monkeypatch.setattr(ocr_router.settings, "tesseract_render_dpi", 151, raising=False)
    monkeypatch.setattr(ocr_router.settings, "paddle_render_dpi", 152, raising=False)

    assert ocr_router._get_engine_render_dpi("deepseek") == 61
    assert ocr_router._get_engine_render_dpi("tesseract") == 151
    assert ocr_router._get_engine_render_dpi("paddle") == 152
    # Unknown defaults to paddle settings
    assert ocr_router._get_engine_render_dpi("unknown") == 152
    # Non-positive DPI is clamped
    monkeypatch.setattr(ocr_router.settings, "paddle_render_dpi", 0, raising=False)
    assert ocr_router._get_engine_render_dpi("paddle") == 150


def test_mark_fallback_reason_sets_once():
    from extraction import ocr_router

    pages = [_mk_page(1, blocks=[_mk_block("x")])]
    # If we did not try at least 2 engines, do nothing
    ocr_router._mark_fallback_reason(pages, ["tesseract"])
    assert "ocr_fallback_reason" not in pages[0].metadata

    # With an engine switch, set the reason
    ocr_router._mark_fallback_reason(pages, ["tesseract", "paddle"], fallback_reason="engine_timeout")
    assert pages[0].metadata["ocr_fallback_reason"] == "engine_timeout"

    # Do not overwrite existing reason
    ocr_router._mark_fallback_reason(pages, ["deepseek", "tesseract"], fallback_reason="dependency_missing")
    assert pages[0].metadata["ocr_fallback_reason"] == "engine_timeout"


def test_run_ocr_engine_unknown_raises_value_error():
    from extraction import ocr_router

    with pytest.raises(ValueError):
        ocr_router._run_ocr_engine(Path("/tmp/x.pdf"), "nope")


def test_try_digital_extract_low_text_density_returns_none(monkeypatch, tmp_path):
    from extraction import ocr_router

    class _FakePage:
        def __init__(self, txt: str):
            self._txt = txt

        def get_text(self, _kind: str):
            return self._txt

    class _FakeDoc:
        page_count = 2

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def load_page(self, i: int):
            return _FakePage("" if i == 0 else "  ")

    class _FakeFitz(types.SimpleNamespace):
        def open(self, _path):
            return _FakeDoc()

    monkeypatch.setitem(sys.modules, "fitz", _FakeFitz())

    pdf = tmp_path / "d.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    assert ocr_router._try_digital_extract(pdf, run_layout_analysis=False) is None


def test_try_digital_extract_high_text_density_runs_parser_and_line_extractor(monkeypatch, tmp_path):
    from extraction import ocr_router

    class _FakePage:
        def __init__(self, txt: str):
            self._txt = txt

        def get_text(self, _kind: str):
            return self._txt

    class _FakeDoc:
        page_count = 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def load_page(self, _i: int):
            return _FakePage("x" * 500)

    class _FakeFitz(types.SimpleNamespace):
        def open(self, _path):
            return _FakeDoc()

    monkeypatch.setitem(sys.modules, "fitz", _FakeFitz())

    pdf = tmp_path / "digital.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    parsed_pages = [_mk_page(1, blocks=[_mk_block("native")])]
    extracted_pages = [_mk_page(1, blocks=[_mk_block("lines")])]

    fake_pdf_parser = types.SimpleNamespace(
        parse_pdf_words_as_lines=lambda *_a, **_k: parsed_pages
    )
    fake_line_extractor = types.SimpleNamespace(
        extract_lines=lambda pages: extracted_pages
    )

    monkeypatch.setitem(sys.modules, "extraction.pdf_parser", fake_pdf_parser)
    monkeypatch.setitem(sys.modules, "extraction.line_extractor", fake_line_extractor)

    out = ocr_router._try_digital_extract(pdf, run_layout_analysis=False)
    assert out is extracted_pages


def test_try_digital_extract_parser_error_falls_back_to_ocr(monkeypatch, tmp_path):
    from extraction import ocr_router

    class _FakePage:
        def get_text(self, _kind: str):
            return "x" * 500

    class _FakeDoc:
        page_count = 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def load_page(self, _i: int):
            return _FakePage()

    class _FakeFitz(types.SimpleNamespace):
        def open(self, _path):
            return _FakeDoc()

    monkeypatch.setitem(sys.modules, "fitz", _FakeFitz())

    pdf = tmp_path / "digital.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    fake_pdf_parser = types.SimpleNamespace(
        parse_pdf_words_as_lines=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("parse failed"))
    )
    monkeypatch.setitem(sys.modules, "extraction.pdf_parser", fake_pdf_parser)

    assert ocr_router._try_digital_extract(pdf, run_layout_analysis=False) is None


def test_normalize_engine_name_strips_and_lowercases():
    from extraction import ocr_router

    assert ocr_router._normalize_engine_name("  TeSsErAcT ") == "tesseract"
    assert ocr_router._normalize_engine_name("") == ""


def test_filter_contained_blocks_removes_wrapper_and_duplicate():
    from extraction import ocr_router

    wrapper = {"bbox": {"x": 0, "y": 0, "width": 100, "height": 100}}
    inner = {"bbox": {"x": 10, "y": 10, "width": 80, "height": 80}}
    inner2 = {"bbox": {"x": 11, "y": 11, "width": 79, "height": 79}}

    out = ocr_router._filter_contained_blocks([wrapper, inner, inner2])
    assert len(out) == 1


def test_apply_scanned_policy_auto_fallback_uses_settings_default_chain(monkeypatch):
    from extraction import ocr_router

    monkeypatch.setattr(ocr_router.settings, "ocr_scanned_default_chain", ["tesseract", "paddle"], raising=False)
    priority, allow = ocr_router._apply_scanned_policy_to_priority(None, policy="auto_fallback")
    assert priority == ["tesseract", "paddle"]
    assert allow is True


def test_is_cuda_and_mps_available_with_fake_torch(monkeypatch):
    from extraction import ocr_router

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

    class _FakeMps:
        @staticmethod
        def is_available():
            return True

    fake_torch = types.SimpleNamespace(cuda=_FakeCuda(), backends=types.SimpleNamespace(mps=_FakeMps()))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert ocr_router.is_cuda_available() is True
    assert ocr_router.is_mps_available() is True


def test_ocr_pdf_multi_no_available_engines_returns_empty(monkeypatch, tmp_path):
    from extraction import ocr_router

    pdf = tmp_path / "none.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    monkeypatch.setattr(ocr_router, "_try_digital_extract", lambda *_a, **_k: None)
    monkeypatch.setattr(ocr_router, "select_ocr_engine", lambda prio: ([], {"paddle": "dependency_missing"}))

    out = ocr_router.ocr_pdf_multi(pdf, prefer_digital=False, run_layout_analysis=False)
    assert out == []


def test_run_deepseek_with_guardrails_budget_and_warnings(monkeypatch, tmp_path):
    from extraction import ocr_router

    monkeypatch.setattr(ocr_router.settings, "deepseek_max_pages_per_doc", 1, raising=False)
    monkeypatch.setattr(ocr_router.settings, "deepseek_render_dpi", 60, raising=False)
    monkeypatch.setattr(ocr_router.settings, "deepseek_ocr_model_path", "/tmp/model", raising=False)

    class _Diagnostics:
        def __init__(self, reason: str):
            self.reason = reason

    class _GuardrailResult:
        def __init__(self, ok: bool, *, blocks=None, warnings=None, engine_meta=None, diagnostics=None):
            self.ok = ok
            self.blocks = list(blocks or [])
            self.warnings = list(warnings or [])
            self.engine_meta = dict(engine_meta or {})
            self.diagnostics = diagnostics

    class _GuardrailViolation(Exception):
        def __init__(self, reason: str):
            super().__init__(reason)
            self.reason = reason

    class _FakeOCR:
        def recognize_page(self, *, image, page_index: int, manifest_page: dict, target_size):
            manifest_page["engine_type"] = "deepseek"
            manifest_page["deepseek_mode"] = "stub"
            return _GuardrailResult(
                True,
                blocks=[_mk_block("ok")],
                warnings=["w1"],
                engine_meta={"elapsed_sec": 0.01, "peak_rss_mb": 10},
            )

    fake_deepseek = types.SimpleNamespace(
        GuardrailViolation=_GuardrailViolation,
        GuardrailResult=_GuardrailResult,
        get_ocr_instance=lambda _path: _FakeOCR(),
    )
    monkeypatch.setitem(sys.modules, "extraction.deepseek_ocr_engine", fake_deepseek)

    class _Rect:
        def __init__(self, w, h):
            self.width = w
            self.height = h

    class _FakePix:
        pass

    class _FakePage:
        def __init__(self, w=600, h=800):
            self.rect = _Rect(w, h)

        def get_pixmap(self, dpi=60, alpha=False):
            assert dpi == 60
            assert alpha is False
            return _FakePix()

    class _FakeDoc:
        page_count = 2

        def __init__(self):
            self._closed = False

        def load_page(self, _i: int):
            return _FakePage()

        def close(self):
            self._closed = True

    fake_doc = _FakeDoc()
    monkeypatch.setitem(sys.modules, "fitz", types.SimpleNamespace(open=lambda _p: fake_doc))

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%fake\n")

    pages = ocr_router._run_deepseek_with_guardrails(pdf)
    assert len(pages) == 2
    assert pages[0].metadata["ocr_engine_used"] == "deepseek"
    assert pages[0].metadata.get("deepseek_warnings") == ["w1"]
    assert pages[1].metadata.get("deepseek_skipped") is True
    assert pages[1].metadata.get("deepseek_skip_reason") == "page_budget_exceeded"


def test_run_deepseek_with_guardrails_rejection_marks_page_failed(monkeypatch, tmp_path):
    from extraction import ocr_router

    monkeypatch.setattr(ocr_router.settings, "deepseek_max_pages_per_doc", 5, raising=False)
    monkeypatch.setattr(ocr_router.settings, "deepseek_render_dpi", 60, raising=False)
    monkeypatch.setattr(ocr_router.settings, "deepseek_ocr_model_path", "/tmp/model", raising=False)

    class _Diagnostics:
        def __init__(self, reason: str):
            self.reason = reason

    class _GuardrailResult:
        def __init__(self, ok: bool, *, diagnostics=None):
            self.ok = ok
            self.blocks = []
            self.warnings = []
            self.engine_meta = {}
            self.diagnostics = diagnostics

    class _GuardrailViolation(Exception):
        def __init__(self, reason: str):
            super().__init__(reason)
            self.reason = reason

    class _FakeOCR:
        def recognize_page(self, *, image, page_index: int, manifest_page: dict, target_size):
            return _GuardrailResult(False, diagnostics=_Diagnostics("too_noisy"))

    fake_deepseek = types.SimpleNamespace(
        GuardrailViolation=_GuardrailViolation,
        GuardrailResult=_GuardrailResult,
        get_ocr_instance=lambda _path: _FakeOCR(),
    )
    monkeypatch.setitem(sys.modules, "extraction.deepseek_ocr_engine", fake_deepseek)

    class _Rect:
        def __init__(self, w, h):
            self.width = w
            self.height = h

    class _FakePage:
        def __init__(self):
            self.rect = _Rect(600, 800)

        def get_pixmap(self, dpi=60, alpha=False):
            return object()

    class _FakeDoc:
        page_count = 1

        def load_page(self, _i: int):
            return _FakePage()

        def close(self):
            return None

    monkeypatch.setitem(sys.modules, "fitz", types.SimpleNamespace(open=lambda _p: _FakeDoc()))

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%fake\n")

    pages = ocr_router._run_deepseek_with_guardrails(pdf)
    assert len(pages) == 1
    assert pages[0].metadata.get("deepseek_failed") is True
    assert pages[0].metadata.get("deepseek_fail_reason") == "too_noisy"


def test_run_deepseek_with_guardrails_violation_raises(monkeypatch, tmp_path):
    from extraction import ocr_router

    monkeypatch.setattr(ocr_router.settings, "deepseek_max_pages_per_doc", 5, raising=False)
    monkeypatch.setattr(ocr_router.settings, "deepseek_render_dpi", 60, raising=False)
    monkeypatch.setattr(ocr_router.settings, "deepseek_ocr_model_path", "/tmp/model", raising=False)

    class _GuardrailResult:
        def __init__(self, ok: bool):
            self.ok = ok
            self.blocks = []
            self.warnings = []
            self.engine_meta = {}
            self.diagnostics = None

    class _GuardrailViolation(Exception):
        def __init__(self, reason: str):
            super().__init__(reason)
            self.reason = reason

    class _FakeOCR:
        def recognize_page(self, *, image, page_index: int, manifest_page: dict, target_size):
            raise _GuardrailViolation("policy")

    fake_deepseek = types.SimpleNamespace(
        GuardrailViolation=_GuardrailViolation,
        GuardrailResult=_GuardrailResult,
        get_ocr_instance=lambda _path: _FakeOCR(),
    )
    monkeypatch.setitem(sys.modules, "extraction.deepseek_ocr_engine", fake_deepseek)

    class _Rect:
        def __init__(self, w, h):
            self.width = w
            self.height = h

    class _FakePage:
        def __init__(self):
            self.rect = _Rect(600, 800)

        def get_pixmap(self, dpi=60, alpha=False):
            return object()

    class _FakeDoc:
        page_count = 1

        def __init__(self):
            self.closed = False

        def load_page(self, _i: int):
            return _FakePage()

        def close(self):
            self.closed = True

    doc = _FakeDoc()
    monkeypatch.setitem(sys.modules, "fitz", types.SimpleNamespace(open=lambda _p: doc))

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%fake\n")

    with pytest.raises(Exception):
        ocr_router._run_deepseek_with_guardrails(pdf)
    assert doc.closed is True
