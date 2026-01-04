from __future__ import annotations


from comparison.models import Line, PageData, TextBlock


def test_extract_lines_no_blocks_is_noop():
    from extraction.line_extractor import extract_lines

    pages = [PageData(page_num=1, width=100.0, height=100.0, blocks=[])]
    out = extract_lines(pages)
    assert out is pages
    assert out[0].lines == []


def test_extract_lines_from_pages_skips_when_lines_already_present():
    from extraction.line_extractor import extract_lines_from_pages

    existing = Line(line_id="l1", bbox={"x": 0, "y": 0, "width": 10, "height": 10}, text="Hello")
    pages = [
        PageData(
            page_num=1,
            width=100.0,
            height=100.0,
            blocks=[TextBlock(text="ignored", bbox={"x": 0, "y": 0, "width": 10, "height": 10})],
            lines=[existing],
        )
    ]

    out = extract_lines_from_pages(pages)
    assert out[0].lines == [existing]


def test_extract_digital_and_ocr_lines_return_empty_when_fitz_missing(monkeypatch, tmp_path):
    from extraction import line_extractor

    monkeypatch.setattr(line_extractor, "fitz", None, raising=True)

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%fake\n")

    assert line_extractor.extract_digital_lines(pdf) == []
    assert line_extractor.extract_ocr_lines(pdf) == []
