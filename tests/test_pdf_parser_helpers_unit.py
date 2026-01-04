from __future__ import annotations

import pytest

from comparison.models import Style


def test_bbox_intersection_area_overlap_and_no_overlap():
    from extraction.pdf_parser import _bbox_intersection_area

    assert _bbox_intersection_area((0, 0, 2, 2), (1, 1, 3, 3)) == 1
    assert _bbox_intersection_area((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    assert _bbox_intersection_area((0, 0, 1, 1), (1, 0, 2, 1)) == 0.0  # touch edge


def test_get_text_flags_is_robust_to_missing_constants():
    from extraction.pdf_parser import _get_text_flags

    class DummyFitz:
        TEXT_PRESERVE_LIGATURES = 8
        TEXT_PRESERVE_WHITESPACE = 16

    assert _get_text_flags(DummyFitz) == 24

    class DummyFitzMissing:
        pass

    assert _get_text_flags(DummyFitzMissing) == 0


def test_style_to_dict_handles_unexpected_size():
    from extraction.pdf_parser import _style_to_dict

    class Weird:
        def __float__(self):  # pragma: no cover
            raise TypeError("no")

    style = Style(font="A", size=Weird(), bold=True, italic=False, color=(1, 2, 3))
    d = _style_to_dict(style)
    assert d["font"] == "A"
    assert d["size"] is None
    assert d["bold"] is True
    assert d["italic"] is False
    assert d["color"] == [1, 2, 3]


@pytest.mark.parametrize(
    "font_in,font_out",
    [
        ("ABCDEF+Calibri", "Calibri"),
        ("CIDFont+F1", "CIDFont"),
        ("RegularFont", "RegularFont"),
        (None, None),
    ],
)
def test_extract_style_from_span_normalizes_font_names(font_in, font_out):
    from extraction.pdf_parser import _extract_style_from_span

    span = {"font": font_in, "size": 12, "flags": 0, "color": 0x112233}
    st = _extract_style_from_span(span)
    assert st.font == font_out
    assert st.size == 12
    assert st.color == (0x11, 0x22, 0x33)


def test_extract_style_from_span_color_list_and_bad_values():
    from extraction.pdf_parser import _extract_style_from_span

    st = _extract_style_from_span({"font": "X", "size": 10, "flags": 0, "color": [1, 2, 3]})
    assert st.color == (1, 2, 3)

    st = _extract_style_from_span({"font": "X", "size": 10, "flags": 0, "color": ["a", 2, 3]})
    assert st.color is None


def test_extract_block_text_concatenates_spans_and_lines():
    from extraction.pdf_parser import _extract_block_text

    block = {
        "lines": [
            {"spans": [{"text": "Hello"}, {"text": " "}, {"text": "world"}]},
            {"spans": [{"text": ""}, {"text": "!"}]},
            {"spans": []},
        ]
    }
    assert _extract_block_text(block) == "Hello world\n!"


def test_build_span_index_skips_invalid_blocks_and_bboxes():
    from extraction.pdf_parser import _build_span_index

    text_dict = {
        "blocks": [
            {"type": 1, "lines": [{"spans": [{"bbox": [0, 0, 1, 1], "font": "A", "size": 9}]}]},
            {"type": 0, "lines": [{"spans": [{"bbox": [0, 0, 1], "font": "A", "size": 9}]}]},
            {"type": 0, "lines": [{"spans": [{"bbox": ["x", 0, 1, 1], "font": "A", "size": 9}]}]},
            {"type": 0, "lines": [{"spans": [{"bbox": [0, 0, 1, 1], "font": "A", "size": 9, "flags": 16}]}]},
        ]
    }

    spans = _build_span_index(text_dict)
    assert len(spans) == 1
    assert spans[0]["bbox"] == (0.0, 0.0, 1.0, 1.0)
    assert isinstance(spans[0]["style"], Style)
    assert spans[0]["style"].bold is True


def test_find_best_span_style_uses_y_pruning_and_picks_best_overlap():
    from extraction.pdf_parser import _find_best_span_style

    s1 = Style(font="A", size=10, bold=False, italic=False, color=None)
    s2 = Style(font="B", size=11, bold=True, italic=False, color=None)

    span_index = [
        {"bbox": (0.0, 100.0, 10.0, 110.0), "style": s1},  # pruned by Y
        {"bbox": (0.0, 0.0, 10.0, 10.0), "style": s1},
        {"bbox": (5.0, 0.0, 15.0, 10.0), "style": s2},  # bigger overlap with target
    ]

    out = _find_best_span_style(span_index, 6, 1, 14, 9)
    assert out is s2

    assert _find_best_span_style([], 0, 0, 1, 1) is None


def test_find_style_for_line_in_cache_selects_span_and_default_fallback():
    from extraction.pdf_parser import _find_style_for_line_in_cache

    cached_lines = [
        (
            0,
            0,
            100,
            10,
            {
                "spans": [
                    {"bbox": [0, 0, 50, 10], "font": "ABCDEF+Calibri", "size": 12, "flags": 16, "color": 0x000000},
                    {"bbox": [60, 0, 100, 10], "font": "Times", "size": 9, "flags": 0, "color": 0xFFFFFF},
                ]
            },
        )
    ]

    st = _find_style_for_line_in_cache(cached_lines, 1, 1, 49, 9)
    assert st.font == "Calibri"
    assert st.bold is True

    st2 = _find_style_for_line_in_cache([], 0, 0, 1, 1)
    assert st2.font is None
    assert st2.bold is False
