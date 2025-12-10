# AI Dokumentų Palyginimo Sistema - Pristatymas

## 📋 Turinys

1. [Sistemos Apžvalga](#sistemos-apžvalga)
2. [Kaip Sistema Veikia](#kaip-sistema-veikia)
3. [Kas Veikia](#kas-veikia)
4. [Kas Dar Reikia Patobulinti](#kas-dar-reikia-patobulinti)
5. [Technologijos ir Modeliai](#technologijos-ir-modeliai)

---

## 🎯 Sistemos Apžvalga

### Paskirtis
AI Dokumentų Palyginimo Sistema yra **local**. Sistema skirta palyginti du PDF dokumentus, skenuotus arba skaitmeninius.

### Pagrindiniai Tikslai
- ✅ **Privatumas**: visa apdorojimo logika vykdoma kompiuteryje
- ✅ **Tikslumas**: nustatyti turinio, formatavimo ir skirtumus
- ✅ **Našumas**: apdoroti iki 60 puslapių per <3s/puslapį
- ✅ **Interaktyvumas**: Gradio sąsaja

---

## ⚙️ Kaip Sistema Veikia

### Architektūra - 3 Pagrindiniai Etapai

```mermaid
flowchart TD
    A[PDF Dokumentas A] --> B[1️⃣ IŠGAVIMAS]
    C[PDF Dokumentas B] --> B
    
    B --> D{Dokumentas skenuotas?}
    D -->|Taip| E[OCR Apdorojimas]
    D -->|Ne| F[PyMuPDF Išgavimas]
    
    E --> G[Layout Analizė]
    F --> G
    
    G --> H[2️⃣ PALYGINIMAS]
    
    H --> I[Puslapių Suderinimas]
    H --> J[Teksto Palyginimas]
    H --> K[Formatavimo Palyginimas]
    H --> L[Vizualinis Palyginimas]
    
    I --> M[3️⃣ VIZUALIZACIJA]
    J --> M
    K --> M
    L --> M
    
    M --> N[Gradio UI]
    N --> O[Sinkronizuotas PDF Viewer]
    N --> P[Skirtumų Navigatorius]
    N --> Q[Heatmap Overlay]
```

### 1️⃣ Extraction

#### Skaitmeniams PDF
- **PyMuPDF (fitz)**: išgauna tekstą, šriftus, spalvas, pozicijas


#### Skenuotiems PDF
**Automatinė OCR Variklio Pasirinkimas**:

```python
# Automatinė Priority Eilė:
1. DeepSeek-OCR (CUDA GPU) - geriausias tikslumas
2. PaddleOCR (CPU/Mac) - greitas ir patikimas
3. Tesseract (atsarginis) - universalus sprendimas
```

**OCR Funkcionalumas**:
- Automatinis teksto atpažinimas iš paveikslėlių
- Bounding box'ų nustatymas kiekvienam teksto blokui
- Struktūros metaduomenų išgavimas (antraštės, lentelės, sąrašai)

#### Layout Analizė
**DocLayout-YOLO** modelis aptinka:
- `title` - Dokumentų antraštės
- `plain text` - Įprasti teksto paragrafai
- `figure` - Paveikslėliai ir diagramos
- `table` - Lentelės
- `formula` - Matematinės formulės
- Ir daugiau (iš viso 10 klasių)

**Našumas**: ~120-160ms per puslapį

#### Antraštės/Poraštės Aptikimas
- Aptinka pasikartojančius elementus dokumento viršuje/apačioje
- Filtruoja juos iš palyginimo, kad sumažintų triukšmą

---

### 2️⃣ Palyginimo Etapas (Comparison)

#### A. Puslapių Suderinimas
```python
# Algoritmas:
1. Skaičiuoja puslapių panašumus (embedding'ai)
2. Nustato geriausią suderinimą tarp dokumentų
3. Aptinka pridėtus/pašalintus puslapius
4. Aptinka layout pokyčius
```

#### B. Teksto Palyginimas
**Sentence Transformer (all-MiniLM-L6-v2)**:
- Generuoja 384-dimensinius embedding'us kiekvienam tekstui
- Skaičiuoja kosinuso panašumą (cosine similarity)
- Threshold: 0.82 (konfigūruojama)

**Aptinka**:
- ✏️ Turinio pakeitimus
- ➕ Pridėjimus
- ➖ Pašalinimus
- 🔄 Semantinius pokyčius

**Character-level diff**:
- Tikslus simbolių lygmens palyginimas
- Paryškina konkrečius pakeitimus tame pačiame sakinyje

#### C. Formatavimo Palyginimas
Lygina:
- 🔤 **Šriftą** (font family, size)
- **B** **Stilių** (bold, italic, underline)
- 🎨 **Spalvą** (RGB skirtumai)
- 📏 **Tarpus** (line height, margins)

#### D. Lentelių Palyginimas
- Išgauna lentelių struktūrą
- Lygina ląsteles (cell-by-cell)
- Aptinka pridėtas/pašalintas eilutes/stulpelius

#### E. Vizualinis Palyginimas
- Pixel-level palyginimas
- Generuoja heatmap'us su skirtumų dengimais
- DPI: 150 (konfigūruojama)

---

### 3️⃣ Vizualizacijos Etapas (Visualization)

#### Gradio UI Komponentai

**1. Failų Įkėlimas**
- Drag-and-drop palaikymas
- Automatinis PDF validavimas

**2. Parametrų Pasirinkimas**
```
📊 Jautrumo Threshold (0.70 - 0.95)
🔍 Scanned Mode (OCR įjungimas)
⚡ Force OCR Mode (priverstinis OCR visiems dokumentams)
🎨 Show Heatmap (vizualiniai skirtumai)
```

**3. Rezultatų Rodymas**

**Gallery View** (numatytasis):
- Side-by-side PDF puslapių peržiūra
- Scroll sync tarp puslapių
- Diff highlight'ai ant puslapių

**Synchronized Viewer**:
- Premium PDF viewer su sinchronizuota navigacija
- Real-time page jumping
- Smooth scrolling

**4. Diff Navigator**
```
📋 Skirtumų Sąrašas:
  ├─ Content Changes (turinys)
  ├─ Formatting Changes (formatavimas)
  ├─ Layout Changes (išdėstymas)
  └─ Visual Changes (vizuali)
  
🔍 Filtrai:
  ☑️ Show Content
  ☑️ Show Formatting
  ☑️ Show Layout
  ☑️ Show Visual
  
⏮️ ⏭️ Previous/Next Navigation
```

**5. Eksportas**
- 📄 JSON formatas (mašinai skaitomas)
- 📑 PDF ataskaita (žmogui skaitomas)

---

## ✅ Kas Veikia

### 🎉 Pilnai Funkcionuojantys Komponentai

#### 1. Modelių Integracija
> [!NOTE]
> Visi modeliai veikia lokaliai, užtikrinant privatumą

| Modelis | Statusas | Paskirtis | Dydis |
|---------|----------|-----------|-------|
| **DeepSeek-OCR** | ✅ Veikia | OCR skenuotiems PDF | ~500MB |
| **Sentence Transformer** | ✅ Veikia | Teksto palyginimas | ~80MB |
| **DocLayout-YOLO** | ✅ Veikia | Layout aptikimas | ~39MB |
| **PaddleOCR** | ✅ Veikia | Atsarginis OCR | Auto-download |
| **Tesseract** | ✅ Veikia | Atsarginis OCR | Sistema |

#### 2. Išgavimo Moduliai
- ✅ PyMuPDF Parser (skaitmeniniai PDF)
- ✅ DeepSeek-OCR Engine (CUDA)
- ✅ PaddleOCR Engine (CPU/Mac) 
- ✅ Tesseract OCR Engine (atsarginis)
- ✅ OCR Router (automatinis variklio pasirinkimas)
- ✅ Layout Analyzer (DocLayout-YOLO)
- ✅ Header/Footer Detector
- ✅ OCR Warmup (background loading)

#### 3. Palyginimo Moduliai
- ✅ Text Comparator (Sentence Transformer)
- ✅ Formatting Comparator
- ✅ Table Comparator (cell-by-cell)
- ✅ Figure Comparator
- ✅ Visual Diff Generator
- ✅ Page Alignment
- ✅ Section Alignment
- ✅ Hierarchical Alignment
- ✅ Diff Classifier
- ✅ Character-level diff

#### 4. Vizualizacija
- ✅ Gradio UI (v6.0.2)
- ✅ PDF Gallery Viewer
- ✅ Synchronized PDF Viewer
- ✅ Diff Navigator
- ✅ Heatmap Overlays
- ✅ Bounding Box Visualization
- ✅ Page Navigation (Prev/Next)
- ✅ Diff Filtering
- ✅ Export JSON/PDF

#### 5. Optimizacijos
- ✅ Model Caching (greitas pakartotinis paleidimas)
- ✅ Background OCR Warmup
- ✅ Batch Similarity Computation
- ✅ Module-level OCR Instance Caching
- ✅ Configurable DPI/Performance Settings

#### 6. Testavimas
> [!IMPORTANT]
> Visi testai praeiti sėkmingai (2025-12-06)

- ✅ Model Loading Tests
- ✅ Extraction Module Tests
- ✅ Comparison Module Tests
- ✅ Full Pipeline Tests
- ✅ App Startup Tests
- ✅ Integration Tests

**Test Results**: Žiūrėti [`TEST_RESULTS.md`](file:///Users/airidas/Documents/KTU/P170M109%20Computational%20Intelligence%20and%20Decision%20Making/project/TEST_RESULTS.md)

---

## 🚧 Kas Dar Reikia Patobulinti

### 1. Našumo Optimizacijos

> [!WARNING]
> OCR apdorojimas gali būti lėtas dideliems dokumentams

#### A. DeepSeek-OCR Optimizacija
**Dabartinė Problema**:
- Pilnų dokumentų OCR apdorojimas gali užtrukti
- DPI nustatymai įtakoja greitį ir tikslumą

**Galimi Sprendimai**:
- [ ] Optimizuoti PDF rendering DPI
- [ ] Implementuoti paralelų puslapių apdorojimą
- [ ] Pridėti progress bar'ą ilgiems procesams
- [ ] Cache OCR rezultatus (išvengti pakartotinio apdorojimo)

#### B. UI Responsiveness
**Dabartinė Problema**:
- Galimas UI "užšalimas" vykdant ilgus OCR procesus

**Galimi Sprendimai**:
- [ ] Async processing su progress updates
- [ ] WebSocket komunikacija real-time feedback
- [ ] Chunk-based PDF processing (po kelias psl.)

### 2. Funkcionalumo Išplėtimai

#### A. Lentelių Palyginimas (Advanced)
**Kas Yra**:
- ✅ Bazinis cell-by-cell palyginimas

**Kas Trūksta**:
- [ ] Table Transformer modelio integracija
- [ ] Automatinis lentelių struktūros išgavimas OCR dokumentams
- [ ] Vizualinis lentelių diff rodymas UI

#### B. Paveikslėlių Palyginimas
**Kas Yra**:
- ✅ Caption palyginimas
- ✅ Bounding box aptikimas

**Kas Trūksta**:
- [ ] Image similarity metrics (perceptual hashing)
- [ ] Vizualinis paveikslėlių diff
- [ ] Chart/graph specific comparison

#### C. Matematinių Formulių Palyginimas
**Kas Trūksta**:
- [ ] LaTeX extraction iš formulių
- [ ] Semantinis formulių palyginimas
- [ ] Formula rendering diff

### 3. UI/UX Patobulinimai

#### A. Synchronized Viewer
**Kas Veikia**:
- ✅ Bazinis sync viewer
- ✅ Page navigation

**Kas Galėtų Būti Geriau**:
- [ ] Smoother scrolling sync
- [ ] Zoom synchronization
- [ ] Highlight diff regions on click
- [ ] Minimize PDF loading states

#### B. Diff Navigator
**Kas Galėtų Būti Geriau**:
- [ ] Diff kategorijų statistika (kiekvienos rūšies diff skaičius)
- [ ] Confidence score visualization (grafikai)
- [ ] Diff grouping pagal puslapius
- [ ] Search per diff aprašymus

#### C. Export Features
**Kas Yra**:
- ✅ JSON export
- ✅ PDF export (bazinis)

**Kas Galėtų Būti Geriau**:
- [ ] HTML export su interaktyvumu
- [ ] Excel export lentelių diff
- [ ] Customizable PDF report templates
- [ ] Diff summary statistics

### 4. Testavimas ir Validacija

#### A. Realių Dokumentų Testavimas
**Kas Trūksta**:
- [ ] Large-scale testing su įvairiais dokumentais
- [ ] Performance benchmarking (įvairiems dokumentų tipams)
- [ ] Edge case handling (corrupt PDFs, very large PDFs)

#### B. Accuracy Metrics
**Kas Trūksta**:
- [ ] Ground truth dataset sukūrimas
- [ ] Precision/Recall metrikų skaičiavimas
- [ ] A/B testing skirtingų threshold'ų

#### C. Unit Testing
**Kas Yra**:
- ✅ Integration tests
- ✅ Module tests

**Kas Galėtų Būti Geriau**:
- [ ] Pytest unit tests kiekvienam moduliui
- [ ] Mock testing modelių
- [ ] CI/CD pipeline setup

### 5. Dokumentacija

**Kas Yra**:
- ✅ README.md
- ✅ models/README.md
- ✅ TEST_RESULTS.md
- ✅ .env.example

**Kas Galėtų Būti Geriau**:
- [ ] API Documentation (docstrings → Sphinx)
- [ ] User Guide (kaip naudoti sistemą)
- [ ] Developer Guide (kaip plėsti sistemą)
- [ ] Video tutorials/demos

### 6. Deployment

**Dabartinis Statusas**:
- ✅ Lokalus paleidimas (`python app.py`)

**Kas Galėtų Būti Geriau**:
- [ ] Docker containerization
- [ ] Docker Compose setup su visais dependencies
- [ ] Cloud deployment guide (AWS/GCP/Azure)
- [ ] Gradio Share link (temporary demo)

---

## 🔧 Technologijos ir Modeliai

### Core Technologies

| Technologija | Versija | Paskirtis |
|--------------|---------|-----------|
| **Python** | 3.9+ | Pagrindinė kalba |
| **Gradio** | 6.0.2 | Web UI framework |
| **PyMuPDF (fitz)** | Latest | PDF manipulation |
| **PyTorch** | Latest | Deep learning backend |
| **Sentence Transformers** | Latest | NLP embeddings |
| **OpenCV** | Latest | Image processing |
| **Pillow** | Latest | Image handling |
| **NumPy** | Latest | Numerical computations |

### AI Modeliai

#### 1. DeepSeek-OCR
```yaml
Modelis: deepseek-ai/deepseek-ocr
Dydis: ~500MB
Framework: HuggingFace Transformers
Device: CUDA (GPU)
Paskirtis: High-accuracy OCR su grounding
Features:
  - Markdown output su bounding boxes
  - Multi-language support
  - Structure recognition
```

#### 2. Sentence Transformer
```yaml
Modelis: sentence-transformers/all-MiniLM-L6-v2
Dydis: ~80MB
Framework: Sentence Transformers
Device: CPU/GPU
Paskirtis: Semantic text similarity
Features:
  - 384-dimensional embeddings
  - Cosine similarity computation
  - Fast batch processing
```

#### 3. DocLayout-YOLO
```yaml
Modelis: juliozhao/DocLayout-YOLO-DocStructBench
Dydis: ~39MB
Framework: Ultralytics YOLO
Device: CPU/GPU
Paskirtis: Document layout analysis
Features:
  - 10 document element classes
  - ~120-160ms inference time
  - Optimized for PDFs
```

#### 4. PaddleOCR
```yaml
Modelis: PaddlePaddle/PaddleOCR v3.x
Dydis: Auto-download
Framework: PaddlePaddle
Device: CPU/Mac M-series
Paskirtis: Fast OCR for non-GPU systems
Features:
  - Multi-language support
  - Text orientation detection
  - Document orientation classification
```

#### 5. Tesseract OCR
```yaml
Engine: Tesseract 5.x
Framework: pytesseract wrapper
Device: CPU
Paskirtis: Fallback OCR engine
Features:
  - Universal language support
  - System-level installation
  - Reliable baseline
```

### Konfigūracija

Visi parametrai konfigūruojami per `.env` failą:

````bash
# Modelių keliai
DEEPSEEK_OCR_MODEL_PATH=models/deepseek-ocr
SENTENCE_TRANSFORMER_MODEL=models/all-MiniLM-L6-v2
YOLO_LAYOUT_MODEL_NAME=models/doclayout_yolo_docstructbench_imgsz1024.pt

# Threshold'ai
TEXT_SIMILARITY_THRESHOLD=0.82
FORMATTING_CHANGE_THRESHOLD=0.1
VISUAL_DIFF_PIXEL_THRESHOLD=30

# Našumo nustatymai
MAX_PAGES=60
SECONDS_PER_PAGE_TARGET=3.0
RENDER_DPI=144
````

---

## 📊 Sistemos Statistika

### Performance Metrics (nuo TEST_RESULTS.md)

| Metrika | Rezultatas | Target | Statusas |
|---------|------------|--------|----------|
| **Similarity Computation** | 0.037s | <0.1s | ✅ Pass |
| **Layout Detection** | 120-160ms | <200ms | ✅ Pass |
| **Model Loading** | ~2-3s | One-time | ✅ Pass |
| **Subsequent Loads** | Instant | Cached | ✅ Pass |

### Model Sizes

```
📦 Total Model Storage: ~620MB
├─ DeepSeek-OCR: ~500MB
├─ Sentence Transformer: ~80MB
└─ DocLayout-YOLO: ~39MB
```

### Test Coverage

```
✅ Model Tests: 100% Pass
✅ Extraction Tests: 100% Pass
✅ Comparison Tests: 100% Pass
✅ Pipeline Tests: 100% Pass
✅ App Startup: 100% Pass
```

---

## 🎯 Santrauka

### Stipriosios Pusės
1. ✅ **Pilnai vietinė** - jokių išorinių API kvietimų
2. ✅ **Modulinė architektūra** - lengva plėsti ir keisti
3. ✅ **Automatinė OCR pasirinkimas** - veikia bet kokiame hardware
4. ✅ **Interaktyvi UI** - Gradio su real-time rezultatais
5. ✅ **Išsamus testavimas** - visi komponentai patikrinti

### Tobulinimo Kryptys
1. 🚧 **Našumo optimizacijos** - greitesnis OCR apdorojimas
2. 🚧 **Advanced features** - lentelių, paveikslėlių, formulių palyginimas
3. 🚧 **UI/UX patobulinimai** - smoother interactions, better visualization
4. 🚧 **Testavimas** - daugiau realių dokumentų testų
5. 🚧 **Deployment** - Docker, cloud setup

### Veikiantis Prototipas
Sistema yra **pilnai funkcionuojantis prototipas** su visais pagrindiniais komponentais:
- ✅ OCR (3 varikliai)
- ✅ Layout Analysis
- ✅ Text/Formatting/Visual Comparison
- ✅ Interactive UI
- ✅ Export Features

**Ready for demonstration and further development! 🚀**

---

## 📚 Naudingos Nuorodos

**Projekto Failai**:
- [README.md](file:///Users/airidas/Documents/KTU/P170M109%20Computational%20Intelligence%20and%20Decision%20Making/project/README.md)
- [TEST_RESULTS.md](file:///Users/airidas/Documents/KTU/P170M109%20Computational%20Intelligence%20and%20Decision%20Making/project/TEST_RESULTS.md)
- [models/README.md](file:///Users/airidas/Documents/KTU/P170M109%20Computational%20Intelligence%20and%20Decision%20Making/project/models/README.md)

**Pagrindiniai Moduliai**:
- [app.py](file:///Users/airidas/Documents/KTU/P170M109%20Computational%20Intelligence%20and%20Decision%20Making/project/app.py) - Entry point
- [gradio_ui.py](file:///Users/airidas/Documents/KTU/P170M109%20Computational%20Intelligence%20and%20Decision%20Making/project/visualization/gradio_ui.py) - UI
- [ocr_router.py](file:///Users/airidas/Documents/KTU/P170M109%20Computational%20Intelligence%20and%20Decision%20Making/project/extraction/ocr_router.py) - OCR routing
- [text_comparison.py](file:///Users/airidas/Documents/KTU/P170M109%20Computational%20Intelligence%20and%20Decision%20Making/project/comparison/text_comparison.py) - Text comparison

---

**Sukurta**: 2025-12-09  
**Versija**: 1.0  
**Sistema veikia**: ✅ http://localhost:7860
