from __future__ import annotations

import unicodedata

from comparison.text_normalizer import (
    NormalizationConfig,
    NormalizedText,
    classify_text_diff,
    compute_diff_metrics,
    compute_page_fingerprint,
    compute_text_fingerprint,
    deduplicate_for_encoding,
    is_formatting_only_diff,
    is_whitespace_only_diff,
    normalize_compare,
    normalize_strict,
    normalize_texts_batch,
)


def test_normalize_strict_nfc_and_newlines():
    # e + combining dot above -> composed if possible
    raw = "e\u0307\n\n\nA"
    out = normalize_strict(raw)
    assert "\n\n\n" not in out
    # NFC should be applied
    assert unicodedata.normalize("NFC", raw).strip().startswith(out.splitlines()[0])


def test_normalize_compare_quotes_dashes_spaces_and_hyphenation():
    cfg = NormalizationConfig.default_ocr()

    raw = "\u201CQuote\u201D  word-\nwrap  A\u00A0B  x \u2014 y  -- z\u200b"
    out = normalize_compare(raw, cfg)

    # Quotes normalized
    assert '"quote"' in out
    # Hyphenation join removes "-\n"
    assert "wordwrap" in out
    # NBSP -> space and collapse whitespace
    assert "a b" in out
    # Dash variants collapse, OCR mode removes spaces around dashes
    assert "x-y" in out
    # Multiple dashes collapse
    assert "- z" not in out  # space handling should be normalized


def test_fingerprints_and_normalizedtext_matches():
    cfg = NormalizationConfig.default_digital()

    a = NormalizedText.from_text("Hello", cfg)
    b = NormalizedText.from_text("hello", cfg)

    assert a.matches(b) is True
    assert compute_text_fingerprint(a.compare_text) == a.fingerprint


def test_page_fingerprint_deterministic():
    cfg = NormalizationConfig.default_digital()

    fp1 = compute_page_fingerprint(["A", "B"], cfg)
    fp2 = compute_page_fingerprint(["a", "b"], cfg)
    assert fp1 == fp2


def test_classify_text_diff_whitespace_formatting_content():
    cfg = NormalizationConfig.default_digital()

    assert classify_text_diff("Same", "Same", cfg) == (False, "identical")

    # Whitespace-only
    assert is_whitespace_only_diff("A  B", "A\nB", cfg) is True

    # Formatting-only (case)
    assert is_formatting_only_diff("Hello", "hello", cfg) is True

    # Content diff
    sig, kind = classify_text_diff("Hello", "Goodbye", cfg)
    assert sig is True
    assert kind == "content"


def test_batch_and_deduplicate_for_encoding():
    cfg = NormalizationConfig.default_digital()

    norms = normalize_texts_batch(["A", "a", "B"], cfg)
    assert len(norms) == 3
    assert norms[0].matches(norms[1])

    unique, mapping = deduplicate_for_encoding(["A", "a", "B"], cfg)
    # Deterministic ordering (sorted)
    assert unique == sorted(unique)
    assert mapping[0] == mapping[1]
    assert mapping[2] != mapping[0]


def test_compute_diff_metrics_counts_and_ratios():
    cfg = NormalizationConfig.default_digital()

    pairs = [
        ("A", "A"),
        ("A  B", "A\nB"),
        ("Hello", "hello"),
        ("Hello", "Goodbye"),
    ]
    m = compute_diff_metrics(pairs, cfg)
    d = m.to_dict()

    assert d["total_texts"] == 4
    assert d["identical_pairs"] == 1
    assert d["whitespace_only_diffs"] == 1
    assert d["formatting_only_diffs"] == 1
    assert d["content_diffs"] == 1
    assert 0.0 <= d["whitespace_only_ratio"] <= 1.0
    assert 0.0 <= d["formatting_only_ratio"] <= 1.0
