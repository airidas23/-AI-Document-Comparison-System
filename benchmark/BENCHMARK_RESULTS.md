# OCR Engine Benchmark Results

**Data:** 2026-01-03  
**Test Environment:** macOS, Python 3.13.4, Apple Silicon

## 1. Benchmark Summary

| Metrika | PyMuPDF | Tesseract | PaddleOCR |
|---------|---------|-----------|-----------|
| **Tipas** | Native PDF | OCR | Deep Learning OCR |
| **Digital PDF laikas** | 0.005s | 0.876s | 16.871s |
| **Scanned PDF laikas** | N/A | 1.233s | 9.388s |
| **Simboliai (digital)** | 1,482 | 1,292 | 1,448 |
| **Blokai (digital)** | 13 | 192 | 34 |
| **Tikslumas** | 100% (baseline) | 87.2% | 97.7% |

## 2. Greičio Palyginimas

### Digital PDF (1 puslapis)
```
PyMuPDF:   0.005s  ████████████████████████████████████████ (bazė)
Tesseract: 0.876s  ██ (175x lėčiau)
PaddleOCR: 16.87s  █ (3374x lėčiau)
```

### Scanned PDF (1 puslapis)
```
Tesseract: 1.233s  ████████████████████████████████████████
PaddleOCR: 9.388s  █████ (7.6x lėčiau)
```

## 3. Tikslumas (Simbolių Išgavimas)

| Engine | Simboliai | % nuo PyMuPDF | Skirtumas |
|--------|-----------|---------------|-----------|
| PyMuPDF (baseline) | 1,482 | 100.0% | — |
| PaddleOCR | 1,448 | 97.7% | -34 |
| Tesseract | 1,292 | 87.2% | -190 |

**Išvados:**
- **PaddleOCR** tiksliau atpažįsta tekstą (97.7% vs 87.2%)
- **Tesseract** prarado ~190 simbolių (13% nuostolis)

## 4. Modelių Charakteristikos

### PyMuPDF
- ✅ **Greičiausias** - 0.005s/psl
- ✅ Natūralus PDF tekstas be OCR
- ❌ Netinka skenuotiems dokumentams
- **Naudojimas:** Digital PDF su įterptu tekstu

### Tesseract
- ✅ Greitas OCR - 1.2s/psl
- ✅ Nemokamas, atviro kodo
- ⚠️ Mažesnis tikslumas (87%)
- ⚠️ Daug smulkių blokų (192 vs 34)
- **Naudojimas:** Greitas OCR kai tikslumas ne kritiškas

### PaddleOCR (PP-OCRv5)
- ✅ **Aukščiausias tikslumas** - 97.7%
- ✅ Geresnė blokų segmentacija (34 logiškai sugrupuoti)
- ❌ **Lėčiausias** - 9-17s/psl
- **Naudojimas:** Kai reikia maksimalaus tikslumo

## 5. Rekomendacijos

| Scenarijus | Rekomenduojamas Engine |
|------------|------------------------|
| Digital PDF (su tekstu) | **PyMuPDF** |
| Scanned PDF (greitas) | **Tesseract** |
| Scanned PDF (tikslus) | **PaddleOCR** |
| Batch processing (>100 psl) | **Tesseract** |
| Akademiniai dokumentai | **PaddleOCR** |

## 6. JSON Rezultatai

```json
{
  "digital_pdf": {
    "pymupdf": {"time": 0.005, "chars": 1482, "blocks": 13},
    "tesseract": {"time": 0.876, "chars": 1292, "blocks": 192},
    "paddle": {"time": 16.871, "chars": 1448, "blocks": 34}
  },
  "scanned_pdf": {
    "tesseract": {"time": 1.233, "chars": 1300, "blocks": 203},
    "paddle": {"time": 9.388, "chars": 1446, "blocks": 34}
  }
}
```

## 7. Vizualizacija

### Greitis (mažiau = geriau)
```
Digital PDF:
PyMuPDF    [█] 0.005s
Tesseract  [█████████████████] 0.876s
PaddleOCR  [█████████████████████████████████████████████████████████████] 16.87s

Scanned PDF:
Tesseract  [█████████████] 1.233s
PaddleOCR  [█████████████████████████████████████████████████████████████████████████████████████████████████████] 9.388s
```

### Tikslumas (daugiau = geriau)
```
PyMuPDF    [████████████████████████████████████████] 100.0%
PaddleOCR  [███████████████████████████████████████] 97.7%
Tesseract  [███████████████████████████████████] 87.2%
```
