from __future__ import annotations


from comparison.models import Diff, PageData, TextBlock


def _block(text: str, *, x: float, y: float, w: float, h: float) -> TextBlock:
    return TextBlock(text=text, bbox={"x": x, "y": y, "width": w, "height": h}, metadata={})


def test_canon_and_tpl_replace_digits():
    from extraction.header_footer_detector import _canon_hf_text, _tpl_hf_text

    assert _canon_hf_text(None) == ""
    assert _tpl_hf_text(None) is None

    canon = _canon_hf_text("Page 12 of 30")
    assert "{#}" in canon
    assert "12" not in canon
    assert "30" not in canon

    tpl = _tpl_hf_text("  Page 12 of 30  ")
    assert tpl == "Page {#} of {#}"


def test_is_page_number_pattern_common_forms():
    from extraction.header_footer_detector import _is_page_number_pattern

    assert _is_page_number_pattern("Page 1") is True
    assert _is_page_number_pattern("1 / 10") is True
    assert _is_page_number_pattern("1 of 10") is True
    assert _is_page_number_pattern("12") is True
    assert _is_page_number_pattern("Appendix A") is False


def test_detect_headers_footers_marks_repeating_text_but_skips_page_numbers(monkeypatch):
    from extraction.header_footer_detector import detect_headers_footers
    from config.settings import settings

    monkeypatch.setattr(settings, "header_region_height_ratio", 0.2, raising=False)
    monkeypatch.setattr(settings, "footer_region_height_ratio", 0.2, raising=False)
    monkeypatch.setattr(settings, "header_footer_repetition_threshold", 0.5, raising=False)

    pages = []
    for p in (1, 2, 3):
        pages.append(
            PageData(
                page_num=p,
                width=200.0,
                height=100.0,
                blocks=[
                    _block("Report Title", x=10, y=5, w=100, h=10),
                    _block(str(p), x=180, y=5, w=10, h=10),
                    _block("Confidential", x=10, y=88, w=100, h=10),
                ],
            )
        )

    out = detect_headers_footers(pages)
    assert set(out.keys()) == {1, 2, 3}

    for p in (1, 2, 3):
        headers, footers = out[p]
        assert any(h.text == "Report Title" for h in headers)
        assert any(f.text == "Confidential" for f in footers)

        title = next(h for h in headers if h.text == "Report Title")
        assert title.is_repeating is True

        page_num = next(h for h in headers if h.text == str(p))
        assert page_num.is_page_number is True
        assert page_num.is_repeating is False


def test_match_items_uses_position_and_text_threshold(monkeypatch):
    from extraction.header_footer_detector import HeaderFooter, _match_header_footer_items
    from config.settings import settings

    a = [
        HeaderFooter(
            text="Header",
            bbox={"x": 10, "y": 5, "width": 100, "height": 10},
            page_num=1,
            is_header=True,
        )
    ]
    b = [
        HeaderFooter(
            text="Header",
            bbox={"x": 12, "y": 6, "width": 100, "height": 10},
            page_num=1,
            is_header=True,
        )
    ]

    monkeypatch.setattr(settings, "header_footer_match_threshold", 0.1, raising=False)
    matched, ua, ub = _match_header_footer_items(a, b)
    assert len(matched) == 1
    assert ua == []
    assert ub == []

    # Now require perfect match and mismatch on text -> nothing should match.
    b2 = [
        HeaderFooter(
            text="Different",
            bbox={"x": 12, "y": 6, "width": 100, "height": 10},
            page_num=1,
            is_header=True,
        )
    ]
    monkeypatch.setattr(settings, "header_footer_match_threshold", 0.99, raising=False)
    matched2, ua2, ub2 = _match_header_footer_items(a, b2)
    assert matched2 == []
    assert len(ua2) == 1
    assert len(ub2) == 1


def test_compare_headers_footers_aggregates_repeating_diffs(monkeypatch):
    from extraction.header_footer_detector import compare_headers_footers
    from config.settings import settings

    monkeypatch.setattr(settings, "header_region_height_ratio", 0.2, raising=False)
    monkeypatch.setattr(settings, "footer_region_height_ratio", 0.2, raising=False)
    monkeypatch.setattr(settings, "header_footer_match_threshold", 0.1, raising=False)

    pages_a = [
        PageData(page_num=1, width=200.0, height=100.0, blocks=[_block("Report 1", x=10, y=5, w=120, h=10)]),
        PageData(page_num=2, width=200.0, height=100.0, blocks=[_block("Report 2", x=10, y=5, w=120, h=10)]),
    ]
    pages_b = [
        PageData(page_num=1, width=200.0, height=100.0, blocks=[_block("Report 9", x=10, y=5, w=120, h=10)]),
        PageData(page_num=2, width=200.0, height=100.0, blocks=[_block("Report 8", x=10, y=5, w=120, h=10)]),
    ]

    # Force page alignment 1->1, 2->2.
    alignment = {1: (1, 1.0), 2: (2, 1.0)}
    diffs = compare_headers_footers(pages_a, pages_b, alignment_map=alignment)

    # Digit templating should collapse both pages into a single aggregated diff.
    assert len(diffs) == 1
    d = diffs[0]
    assert isinstance(d, Diff)
    assert d.metadata.get("header_footer_change") == "header"
    assert d.metadata.get("aggregated") is True
    assert d.metadata.get("pages") == [1, 2]

    occ = d.metadata.get("occurrences")
    assert isinstance(occ, list)
    assert len(occ) == 2
