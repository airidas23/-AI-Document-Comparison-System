from __future__ import annotations

from comparison.models import Diff


def _diff(
    *,
    change_type: str = "content",
    diff_type: str = "modified",
    old: str | None = "a",
    new: str | None = "b",
    md: dict | None = None,
    confidence: float = 0.9,
) -> Diff:
    return Diff(
        page_num=1,
        diff_type=diff_type,  # type: ignore[arg-type]
        change_type=change_type,  # type: ignore[arg-type]
        old_text=old,
        new_text=new,
        confidence=confidence,
        metadata=dict(md or {}),
    )


def test_ensure_subtype_sets_non_content_subtypes():
    from comparison.diff_classifier import _ensure_subtype

    d = _diff(change_type="content", md={"header_footer_change": "header"})
    _ensure_subtype(d)
    assert d.metadata["subtype"] == "header_footer:header"

    d = _diff(change_type="content", md={"table_change": "table_added"})
    _ensure_subtype(d)
    assert d.metadata["subtype"].startswith("table:")

    d = _diff(change_type="content", md={"figure_change": "figure_added"})
    _ensure_subtype(d)
    assert d.metadata["subtype"].startswith("figure:")

    d = _diff(change_type="formatting", md={"formatting_type": "font_size", "scope": "word"})
    _ensure_subtype(d)
    assert d.metadata["subtype"] == "format:font_size:word"

    d = _diff(change_type="layout", md={"layout_shift": True})
    _ensure_subtype(d)
    assert d.metadata["subtype"] == "layout_shift"

    d = _diff(change_type="visual", md={})
    _ensure_subtype(d)
    assert d.metadata["subtype"] == "visual"


def test_classify_single_diff_basic_rules():
    from comparison.diff_classifier import _classify_single_diff

    # Case-only -> formatting
    d = _diff(old="Hello", new="hello")
    out = _classify_single_diff(d)
    assert out.change_type == "formatting"
    assert out.metadata["subtype"] == "case"

    # Whitespace-only -> formatting
    d = _diff(old="a b", new="a  b")
    out = _classify_single_diff(d)
    assert out.change_type == "formatting"
    assert out.metadata["subtype"] == "whitespace"

    # Punctuation-only -> formatting
    d = _diff(old="Hello.", new="Hello!")
    out = _classify_single_diff(d)
    assert out.change_type == "formatting"
    assert out.metadata["subtype"] == "punctuation"

    # Number format -> formatting
    d = _diff(old="1,000", new="1000")
    out = _classify_single_diff(d)
    assert out.change_type == "formatting"
    assert out.metadata["subtype"] == "number_format"

    # Character change -> content
    d = _diff(old="cat", new="cot")
    out = _classify_single_diff(d)
    assert out.change_type == "content"
    assert out.metadata["subtype"] == "character_change"


def test_apply_ocr_adjustments_sets_expected_flags(monkeypatch):
    from comparison.diff_classifier import _apply_ocr_adjustments

    d = _diff(change_type="formatting", old="a", new="b", md={"is_ocr": True, "subtype": "punctuation"}, confidence=0.95)
    out = _apply_ocr_adjustments(d)
    assert out.confidence <= 0.3
    assert out.metadata["ocr_reliability"] == "low"
    assert out.metadata["ocr_severity"] == "low"


def test_filter_ocr_noise_diffs_filters_expected_cases(monkeypatch):
    from comparison.diff_classifier import _filter_ocr_noise_diffs
    from config.settings import settings
    import utils.text_normalization as tn

    # Enable all OCR noise filters.
    monkeypatch.setattr(settings, "ocr_ignore_punctuation_diffs", True, raising=False)
    monkeypatch.setattr(settings, "ocr_ignore_whitespace_diffs", True, raising=False)
    monkeypatch.setattr(settings, "ocr_ignore_case_diffs", True, raising=False)
    monkeypatch.setattr(settings, "ocr_min_change_chars", 2, raising=False)
    monkeypatch.setattr(settings, "ocr_min_change_ratio", 0.02, raising=False)

    # Ensure significance computation path is covered when metadata is missing.
    monkeypatch.setattr(
        tn,
        "compute_ocr_change_significance",
        lambda *_a, **_k: {"changed_chars": 1, "change_ratio": 0.001},
        raising=False,
    )

    keep = _diff(md={"is_ocr": False, "subtype": "punctuation"})

    punct = _diff(change_type="formatting", md={"is_ocr": True, "subtype": "punctuation"})
    ws = _diff(change_type="formatting", md={"is_ocr": True, "subtype": "whitespace"})
    case = _diff(change_type="formatting", md={"is_ocr": True, "subtype": "case"})

    # OCR formatting diff of spacing should be filtered
    spacing = _diff(change_type="formatting", md={"is_ocr": True, "subtype": "format:spacing", "formatting_type": "spacing"})

    # OCR layout diffs should be filtered unless major shift
    layout = _diff(change_type="layout", md={"is_ocr": True, "subtype": "layout_shift"})

    # OCR character_change diff filtered by thresholds, metadata absent triggers significance computation.
    char = _diff(change_type="content", old="abc", new="abd", md={"is_ocr": True, "subtype": "character_change"})

    out = _filter_ocr_noise_diffs([keep, punct, ws, case, spacing, layout, char])
    assert out == [keep]


def test_classify_diffs_with_aggressive_filter(monkeypatch):
    from comparison.diff_classifier import classify_diffs
    from config.settings import settings

    monkeypatch.setattr(settings, "ocr_aggressive_noise_filter", True, raising=False)
    monkeypatch.setattr(settings, "ocr_ignore_punctuation_diffs", True, raising=False)

    d1 = _diff(old="x.", new="x!", md={"is_ocr": True})
    d2 = _diff(old="cat", new="cot", md={"is_ocr": True})

    out = classify_diffs([d1, d2])
    # punctuation diff filtered, character change kept
    assert len(out) == 1
    assert out[0].metadata.get("subtype") == "character_change"


def test_generate_description_and_summary():
    from comparison.diff_classifier import _generate_description, get_diff_summary

    d = _diff(
        change_type="formatting",
        diff_type="modified",
        old="a",
        new="b",
        md={"formatting_type": "font_size", "old_size": 10.0, "new_size": 12.0, "scope": "word", "word_text": "X"},
    )
    _generate_description(d)
    assert "Font size changed" in d.metadata.get("description", "")

    d_ocr_low = _diff(change_type="formatting", md={"is_ocr": True, "subtype": "punctuation", "ocr_severity": "low", "ocr_reliability": "low"})
    summary = get_diff_summary([d, d_ocr_low])
    assert summary["total"] == 2
    assert summary["ocr_stats"]["ocr_diffs"] == 1
    assert summary["ocr_stats"]["ocr_low_severity"] == 1
