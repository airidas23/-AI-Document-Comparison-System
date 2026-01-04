# PDF Dokumentų Palyginimo Sistemos – Testavimo Ataskaita (2026-01-04)

## 1. Santrauka (Executive Summary)
- Testai: `pytest` (full suite) – **487 passed**, **17 skipped**, **0 failed** (trukmė: **48.47s**).
- Golden (10 variacijų): **Precision 0.9714**, **Recall 0.8848**, **F1 0.9227**.
- Našumas (golden): **avg 1.8525 s/page**, **p95 1.9355 s/page**.
- `comparison/hierarchical_alignment.py` coverage (unit): **82%**.

## 2. Aplinka
- OS: macOS
- Python: 3.13.4
- Pytest: 9.0.1
- Pluginai: `pytest-cov`

## 3. Vykdyti testai ir komandos

### 3.1 Pilnas testų rinkinys
Komanda:
```bash
.venv/bin/python -m pytest
```
Rezultatas:
- 504 testai surinkti
- 487 passed, 17 skipped, 0 failed
- 48.47s

### 3.2 `hierarchical_alignment` coverage matavimas
Komanda:
```bash
.venv/bin/python -m pytest \
  --cov=comparison.hierarchical_alignment \
  --cov-report=term-missing \
  tests/test_hierarchical_alignment_unit.py
```
Rezultatas:
- `comparison/hierarchical_alignment.py`: **82%** (305 statements, 54 missed)

### 3.3 Bendras coverage (comparison + extraction)
Komanda:
```bash
.venv/bin/python -m pytest \
  --cov=comparison --cov=extraction \
  --cov-report=xml --cov-report=term
```
Rezultatas:
- TOTAL coverage: **80%** (5896 statements, 952 missed, 2452 branches, 456 partial; generuotas `coverage.xml`)

## 4. Kokybinės metrikos (P/R/F1)

### 4.1 Golden rinkinys
Šaltinis: [tests/golden_results.json](../tests/golden_results.json)

- Vidurkiai (10 variacijų):
  - Precision: **0.9714**
  - Recall: **0.8848**
  - F1: **0.9227**

- F1 pagal kategorijas (vidurkiai per 10 variacijų):
  - Content: **0.95**
  - Formatting: **0.75**
  - Layout: **0.8333**
  - Visual: **1.0**

### 4.2 Sintetinis rinkinys (10 variacijų)
Šaltinis: [data/synthetic/dataset/evaluation_results.json](../data/synthetic/dataset/evaluation_results.json)

- Vidurkiai:
  - Precision: **0.8527**
  - Recall: **0.5992**
  - F1: **0.6810**

Pastaba: šis rinkinys turi didesnį FP kiekį (detected_count dažnai > ground_truth_count), todėl F1/recall žemesni.

## 5. Našumo metrikos (Latency)

### 5.1 Golden rinkinio latency
Šaltinis: [tests/golden_results.json](../tests/golden_results.json)
- Avg time/page: **1.8525 s**
- P95 time/page: **1.9355 s**

### 5.2 OCR benchmark (engine palyginimas)
Šaltinis: [benchmark/benchmark_results.json](../benchmark/benchmark_results.json)

- Digital PDF:
  - PyMuPDF: **0.005 s**
  - Tesseract: **0.876 s**
  - Paddle: **16.871 s**
- Scanned PDF:
  - Tesseract: **1.233 s**
  - Paddle: **9.388 s**

## 6. Atitikimas slenksčiams (DoD / 6.2)

| Metrika | Tikslas | Minimalus | Faktas | Statusas | Įrodymai |
|---|---:|---:|---:|---|---|
| Precision (bendras) | ≥ 0.85 | 0.80 | 0.9714 | PASS | `tests/golden_results.json` |
| Recall (bendras) | ≥ 0.80 | 0.75 | 0.8848 | PASS | `tests/golden_results.json` |
| F1 (bendras) | ≥ 0.85 | 0.80 | 0.9227 | PASS | `tests/golden_results.json` |
| F1 (formatavimas) | ≥ 0.80 | 0.75 | 0.75 | MIN ONLY | `tests/golden_results.json` |
| Latency (per page) | < 3.0s | 5.0s | p95 1.9355s | PASS | `tests/golden_results.json` |
| IoU (BBox) | > 0.80 | 0.70 | N/A | N/A | Nėra agreguotos IoU metrikos artefaktuose |
| Alignment (puslapiai) | > 95% | 90% | N/A | N/A | Golden rinkinys šiuo metu 1 psl. poros (nėra multi-page GT) |
| WER/CER | <5% / <2% | 10% / 5% | N/A | N/A | WER/CER neskaičiuojama šiame eval pipeline |
| Memory peak | < 4 GB | 6 GB | N/A | N/A | Nėra `memory_profiler` matavimo artefaktų |
| Code coverage (comparison+extraction) | ≥ 80% | N/A | 80% | PASS | `coverage.xml` |
| Code coverage (`hierarchical_alignment`) | ≥ 80% | N/A | 82% | PASS | `pytest --cov=comparison.hierarchical_alignment` |
| Crash rate | 0% | 1% | 0 (pytest) | PASS | `pytest` full suite be failų |

## 7. Praleisti testai (skipped) ir priežastys
Pagal `pytest -ra` suvestinę:
- DeepSeek integraciniai testai praleisti pagal dizainą (reikia realaus modelio / `RUN_DEEPSEEK_OCR=1`).
- OCR parity testai praleisti, nes trūksta testinių PDF failų.
- Doclayout / YOLO testai praleisti, nes trūksta testinių PDF.
- OCR BBox vizualizacijos testas praleistas (reikia `RUN_OCR_BBOX_VIZ=1`).

## 8. Pastabos ir rekomendacijos
1. **Formatting F1 = 0.75**: atitinka minimalų slenkstį, bet ne Sprint 2 tikslą (0.80). Rekomenduojama išplėsti golden rinkinį su daugiau formatavimo variacijų (case/punctuation/whitespace/font).
2. **CI stabilumas**: DeepSeek laikyti už `RUN_DEEPSEEK_OCR=1` (kaip dabar), kad CI nelūžtų.
3. **IoU / WER / CER / Memory**: jei tai būtina atsiskaitymui, reikės papildomų matavimų/artefaktų (pvz. `memory_profiler` ir OCR ground-truth tekstų).

