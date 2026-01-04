# Tesseract benchmark (pytesseract)

This repo includes a small, reproducible benchmark harness for systematically tuning:
- preprocessing (OpenCV)
- Tesseract config (`--psm`, `--oem`, `-l`)
- PDF rasterization DPI

It writes run artifacts into `runs/tesseract/<run_id>/`.

## Install prerequisites (macOS)

- Tesseract engine:
  - `brew install tesseract`
- Poppler (for `pdf2image`):
  - `brew install poppler`

Python deps are in `requirements.txt`.

## Dataset layout

The benchmark discovers samples in a flexible layout.

### Option A (folder sample)

```
dataset/
  scans/
    sample_01/
      input.pdf
      gt.txt
```

### Option B (file pair)

```
dataset/
  easy/
    sample_01.png
    sample_01.gt.txt
```

Ground-truth is optional (if missing, it still runs confidence/time/bbox debugging).

## Run baseline + DataCamp-style A–F preprocessing

```
python scripts/run_tesseract_bench.py \
  --dataset dataset \
  --preprocess none,grayscale,gray_denoise,gray_binarize,gray_denoise_binarize,gray_sharpen \
  --psm 6 \
  --oem 3 \
  --lang eng \
  --dpi 200,300,400 \
  --emit-bbox
```

## Grid search PSM/OEM/LANG (after you pick preprocessing)

```
python scripts/run_tesseract_bench.py \
  --dataset dataset \
  --preprocess gray_denoise_binarize \
  --psm 3,4,6,11 \
  --oem 1,3 \
  --lang lit,eng,lit+eng
```

## Searchable PDF output

This uses `pytesseract.image_to_pdf_or_hocr(..., extension="pdf")` and merges page PDFs.

```
python scripts/run_tesseract_bench.py \
  --dataset dataset \
  --preprocess gray_binarize \
  --psm 6 \
  --oem 3 \
  --lang eng \
  --dpi 300 \
  --emit-searchable-pdf
```

## Outputs

Each run folder contains:
- `meta.json` (platform, versions, args, pip freeze)
- `scores.csv` (per-page rows; PDF also gets an `ALL` row if GT exists)
- `pred/<sample_id>/*.txt`
- `artifacts/bbox/<sample_id>/*.png` (if enabled)
- `artifacts/searchable/<sample_id>/*.pdf` (if enabled)
