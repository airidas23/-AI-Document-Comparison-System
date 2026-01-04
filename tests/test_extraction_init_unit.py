from __future__ import annotations

import types
from pathlib import Path

import pytest

from comparison.models import PageData, TextBlock


def _page(page_num: int, text: str, *, engine: str = "paddle") -> PageData:
    return PageData(
        page_num=page_num,
        width=200.0,
        height=200.0,
        blocks=[TextBlock(text=text, bbox={"x": 1, "y": 2, "width": 10, "height": 5})] if text else [],
        metadata={"ocr_engine_used": engine},
    )


def test_compute_text_quality_score_empty_short():
    from extraction import compute_text_quality_score

    assert compute_text_quality_score("")["quality_score"] == 0.0
    assert compute_text_quality_score("  hi ")["quality_score"] == 0.0


def test_compute_text_quality_score_lithuanian_bonus_and_cid_penalty():
    from extraction import compute_text_quality_score

    good = compute_text_quality_score("Ąžuolas ėjo į mokyklą ir žaidė kieme su draugais.")
    assert good["char_count"] > 10
    assert good["lithuanian_ratio"] > 0.0
    assert 0.0 <= good["quality_score"] <= 1.0

    bad = compute_text_quality_score("(cid:123)(cid:456) (cid:789) (cid:101) (cid:111)")
    assert bad["cid_ratio"] > 0.0
    assert bad["quality_score"] < 0.6


def test_looks_like_repetition_heuristics():
    from extraction import _looks_like_repetition

    assert _looks_like_repetition("too short") is False

    # Dominant token repetition
    text = ("word " * 260).strip()
    assert _looks_like_repetition(text) is True

    # 6 identical tokens in a row
    mixed = "a b c d e f " * 40 + "same same same same same same " + "tail " * 40
    assert _looks_like_repetition(mixed) is True


def test_ocr_is_safe_to_replace_native_basic_branches(monkeypatch):
    from extraction import _ocr_is_safe_to_replace_native
    from config.settings import settings

    # native empty
    ok, reason, overlap = _ocr_is_safe_to_replace_native("", "some ocr text", "paddle")
    assert ok is True
    assert reason == "native_empty"
    assert overlap == 1.0

    # ocr empty
    ok, reason, overlap = _ocr_is_safe_to_replace_native("native text " * 10, "", "paddle")
    assert ok is False
    assert reason == "ocr_empty"
    assert overlap == 0.0

    # low overlap
    monkeypatch.setattr(settings, "hybrid_ocr_min_word_overlap", 0.8, raising=False)
    ok, reason, overlap = _ocr_is_safe_to_replace_native(
        "this is native content about statistics and models",
        "totally different words here with enough length to pass the empty threshold",
        "paddle",
    )
    assert ok is False
    assert reason in {"ocr_low_overlap", "no_words"}
    assert 0.0 <= overlap <= 1.0


def test_ocr_is_safe_to_replace_native_repetition_reject(monkeypatch):
    from extraction import _ocr_is_safe_to_replace_native
    from config.settings import settings

    monkeypatch.setattr(settings, "hybrid_ocr_reject_repetition", True, raising=False)

    native = "lorem ipsum dolor sit amet " * 40
    ocr = ("loop " * 260).strip()

    ok, reason, overlap = _ocr_is_safe_to_replace_native(native, ocr, "deepseek")
    assert ok is False
    assert reason == "ocr_repetition"
    assert 0.0 <= overlap <= 1.0


def test_merge_extraction_results_metadata_and_block_choice(monkeypatch):
    from extraction import _merge_extraction_results
    from config.settings import settings

    # Make gate permissive to ensure replacement branch is reachable.
    monkeypatch.setattr(settings, "hybrid_ocr_min_word_overlap", 0.1, raising=False)
    monkeypatch.setattr(settings, "hybrid_ocr_max_length_ratio", 10.0, raising=False)
    monkeypatch.setattr(settings, "hybrid_ocr_reject_repetition", False, raising=False)

    native_pages = [
        _page(1, "native text " * 30, engine="paddle"),
        _page(2, "native short", engine="paddle"),
    ]
    ocr_pages = [
        _page(1, "native text " * 60, engine="paddle"),  # clearly longer -> replace
        _page(2, "ocr", engine="paddle"),  # too short -> should not replace
    ]

    merged = _merge_extraction_results(native_pages, ocr_pages)
    assert [p.page_num for p in merged] == [1, 2]

    p1 = merged[0]
    assert p1.metadata["extraction_method"] == "hybrid"
    assert p1.metadata["native_blocks"] == 1
    assert p1.metadata["ocr_blocks"] == 1
    assert "hybrid_ocr_decision" in p1.metadata
    # replaced blocks
    assert sum(len(b.text) for b in p1.blocks) > sum(len(b.text) for b in native_pages[0].blocks)

    p2 = merged[1]
    assert p2.metadata["extraction_method"] == "hybrid"
    # should keep native blocks due to short OCR
    assert sum(len(b.text) for b in p2.blocks) == sum(len(b.text) for b in native_pages[1].blocks)


def test_extract_pdf_routes_to_ocr_or_native(monkeypatch, tmp_path: Path):
    from extraction import extract_pdf
    from config.settings import settings

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%fake\n")

    called = {"ocr": 0, "native": 0}

    def fake_ocr(path, *, engine_priority=None, run_layout_analysis=True):
        called["ocr"] += 1
        return [_page(1, "ocr result", engine="paddle")]

    def fake_native(path, *, run_layout_analysis=True):
        called["native"] += 1
        return [_page(1, "native result", engine="paddle")]

    monkeypatch.setattr("extraction.ocr_pdf_multi", fake_ocr)
    monkeypatch.setattr("extraction.parse_pdf_words_as_lines", fake_native)

    # digital path
    monkeypatch.setattr("extraction._is_scanned_pdf", lambda p: False)
    monkeypatch.setattr(settings, "ocr_enhancement_mode", "auto", raising=False)
    monkeypatch.setattr(settings, "use_ocr_for_all_documents", False, raising=False)
    pages = extract_pdf(pdf, force_ocr=False, run_layout_analysis=False)
    assert pages[0].blocks[0].text.startswith("native")
    assert called["native"] == 1

    # scanned path
    monkeypatch.setattr("extraction._is_scanned_pdf", lambda p: True)
    pages = extract_pdf(pdf, force_ocr=False, run_layout_analysis=False)
    assert pages[0].blocks[0].text.startswith("ocr")
    assert called["ocr"] == 1

    # forced ocr
    pages = extract_pdf(pdf, force_ocr=True, run_layout_analysis=False)
    assert pages[0].blocks[0].text.startswith("ocr")
    assert called["ocr"] == 2


def test_is_scanned_pdf_importerror_branch(monkeypatch, tmp_path: Path):
    from extraction import _is_scanned_pdf

    pdf = tmp_path / "b.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%fake\n")

    # Simulate missing fitz
    monkeypatch.setitem(__import__("sys").modules, "fitz", None)

    # Force ImportError by making import fail
    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "fitz":
            raise ImportError("no fitz")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(__import__("builtins"), "__import__", fake_import)

    assert _is_scanned_pdf(pdf) is False


def test_is_scanned_pdf_text_sampling_and_garbage_detection(monkeypatch, tmp_path: Path):
    from extraction import _is_scanned_pdf
    from config.settings import settings

    pdf = tmp_path / "c.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%fake\n")

    class _FakePage:
        def __init__(self, text: str):
            self._text = text

        def get_text(self, mode: str):
            assert mode == "text"
            return self._text

    class _FakeDoc:
        def __init__(self, texts):
            self._pages = [_FakePage(t) for t in texts]

        def __len__(self):
            return len(self._pages)

        def __getitem__(self, i):
            return self._pages[i]

        def close(self):
            return None

    # 1 good page, 1 garbage page (CID), sample_size=2 => ratio 0.5
    texts = [
        "This is a valid text layer with enough letters and numbers 1234." * 2,
        "(cid:123)(cid:456)" * 50,
    ]

    fake_fitz = types.SimpleNamespace(open=lambda path: _FakeDoc(texts))

    # Ensure settings paths that sanitize sample/min chars
    monkeypatch.setattr(settings, "scan_detection_sample_pages", -1, raising=False)
    monkeypatch.setattr(settings, "scan_detection_page_text_min_chars", -1, raising=False)
    monkeypatch.setattr(settings, "scan_detection_min_good_pages_ratio", 0.9, raising=False)

    monkeypatch.setitem(__import__("sys").modules, "fitz", fake_fitz)

    # With min_good_pages_ratio=0.9 and ratio=0.5 => scanned
    assert _is_scanned_pdf(pdf) is True
