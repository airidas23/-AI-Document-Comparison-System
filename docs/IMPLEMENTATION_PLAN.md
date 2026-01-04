# Akademinio Tikslumo PDF Palyginimo Implementacijos Planas

> **Tikslas:** Sumažinti klaidingai teigiamus skirtumus („193 fake diffs" → <10),  
> išsaugoti lietuviškas raides, atskirti formatavimo ir turinio pakeitimus,  
> gauti cell-level table diff'us.

---

## 1. Architektūros Apžvalga

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INPUT: PDF A, PDF B                        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    EXTRACTION LAYER (per page)                      │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────┐  │
│  │  is_digital()   │  │ parse_pdf_words  │  │   ocr_pdf_multi    │  │
│  │  (text ratio)   │──│  _as_lines()     │──│   (PaddleOCR etc)  │  │
│  └─────────────────┘  └──────────────────┘  └────────────────────┘  │
│                              │                        │              │
│                              └──────────┬─────────────┘              │
│                                         ▼                            │
│                          normalize_extraction()                      │
│                    (NFC, NBSP→space, zero-width strip)              │
│                                         │                            │
│                                         ▼                            │
│                        analyze_layout() [DocLayout-YOLO]            │
│                    tables[], figures[], formulas[], paragraphs[]    │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         ALIGNMENT LAYER                             │
│  ┌──────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │  align_pages()   │  │ align_sections()│  │ detect_layout_shift │ │
│  │  (text hash)     │──│  (paragraph IDs)│──│   (reflow vs move)  │ │
│  └──────────────────┘  └─────────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTI-LAYER DIFF ENGINE                          │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │   TEXT DIFF     │  │  FORMATTING     │  │     LAYOUT          │  │
│  │  (line→word→    │  │    DIFF         │  │      DIFF           │  │
│  │   char level)   │  │ (font/size/clr) │  │  (move/reflow)      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │   TABLE DIFF    │  │   FIGURE DIFF   │  │   FORMULA DIFF      │  │
│  │  (cell-level +  │  │  (pHash + OCR   │  │    (LaTeX OCR +     │  │
│  │   borders)      │  │   caption)      │  │     similarity)     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          FUSION LAYER                               │
│   fuse_diffs() → deduplicate by IoU → triangulation confidence     │
│   classify_diffs() → content | formatting | layout | visual        │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           OUTPUT                                    │
│   ComparisonResult → JSON export → PDF overlay → Gradio UI         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Extraction Strategy (A sritis)

### 2.1 Digital vs OCR Aptikimas

**Esamas sprendimas:** `extraction/__init__.py::_is_scanned_pdf()`  
**Problema:** Tikrina tik ar yra teksto, bet ne teksto kokybę.

**Patobulinimas:**
```python
# extraction/__init__.py - naujas _text_quality_score()
def _text_quality_score(text: str) -> float:
    """
    Įvertina teksto kokybę (0-1).
    - Lietuviškų simbolių proporcija (ąčęėįšųūž)
    - Žodžio ilgio vidurkis (>3 = gerai)
    - Replacement char (�) proporcija (mažiau = geriau)
    - CID glyph proporcija (mažiau = geriau)
    """
    if not text or len(text) < 50:
        return 0.0
    
    # Lithuanian letter bonus
    lt_chars = set("ąčęėįšųūžĄČĘĖĮŠŲŪŽ")
    lt_count = sum(1 for c in text if c in lt_chars)
    lt_ratio = lt_count / len(text)
    
    # Word length check
    words = text.split()
    avg_word_len = sum(len(w) for w in words) / max(1, len(words))
    
    # Replacement chars
    replacement_ratio = text.count("�") / len(text)
    
    # CID glyphs (font mapping failures)
    cid_matches = re.findall(r"\(cid:\d+\)", text)
    cid_ratio = (len(cid_matches) * 8) / len(text)
    
    score = 1.0
    score -= replacement_ratio * 5.0  # Heavy penalty
    score -= cid_ratio * 3.0
    score += lt_ratio * 0.2  # Bonus for Lithuanian
    score += min(avg_word_len / 10, 0.3)
    
    return max(0.0, min(1.0, score))
```

### 2.2 Canonical Line Blocks su Word Metadata

**Esamas sprendimas:** `extraction/line_extractor.py::extract_document_lines()`  
**Tinka:** Jau turi `Token` su bbox, confidence, text.

**Patobulinimas:** Pridėti style informaciją iš spans:
```python
# extraction/line_extractor.py - papildyti Token
@dataclass
class Token:
    token_id: str
    bbox: dict  # {x, y, width, height} normalized 0-1
    text: str
    confidence: float = 1.0
    # NAUJI LAUKAI:
    style: Optional[Style] = None  # font, size, flags
    span_index: int = -1           # PyMuPDF span index for style lookup
```

### 2.3 Normalizacija

**Esamas sprendimas:** `utils/text_normalization.py::normalize_text()`  
**Problema:** Tik lowercase + whitespace, nėra NBSP/zero-width.

**Patobulinimas:**
```python
# utils/text_normalization.py
import unicodedata

def normalize_text_full(text: str, *, preserve_case: bool = False) -> str:
    """
    Pilna normalizacija akademiniam palyginimui.
    
    1. NFC normalization (composite Lithuanian chars)
    2. NBSP → space
    3. Zero-width characters removal
    4. Soft hyphen removal
    5. Multiple whitespace → single space
    6. Optional lowercase
    """
    if not text:
        return ""
    
    # NFC - kritiškai svarbu lietuviškoms raidėms
    text = unicodedata.normalize("NFC", text)
    
    # NBSP ir kiti special spaces → normalus tarpas
    text = text.replace("\u00A0", " ")  # NBSP
    text = text.replace("\u202F", " ")  # Narrow NBSP
    text = text.replace("\u2007", " ")  # Figure space
    text = text.replace("\u2009", " ")  # Thin space
    
    # Zero-width removal
    text = text.replace("\u200B", "")   # Zero-width space
    text = text.replace("\u200C", "")   # Zero-width non-joiner
    text = text.replace("\u200D", "")   # Zero-width joiner
    text = text.replace("\uFEFF", "")   # BOM/ZWNBSP
    
    # Soft hyphen
    text = text.replace("\u00AD", "")
    
    # Whitespace collapse
    text = " ".join(text.split())
    
    if not preserve_case:
        text = text.lower()
    
    return text.strip()
```

---

## 3. Multi-Layer Diff Algorithm (B sritis)

### 3.1 Text Diff (line→word→char)

**Esamas sprendimas:** `comparison/line_comparison.py::compare_lines()`  
**Problema:** `_compute_word_level_highlight()` naudoja SequenceMatcher, bet negrąžina char-level info.

**Patobulinimas:**
```python
# comparison/line_comparison.py
from difflib import SequenceMatcher
from typing import List, Tuple

@dataclass
class WordDiff:
    """Vieno žodžio pakeitimo informacija."""
    word_a: str
    word_b: str
    bbox_a: Optional[dict]  # bbox dokumente A
    bbox_b: Optional[dict]  # bbox dokumente B
    change_type: str        # "same" | "modified" | "added" | "deleted"
    char_diffs: List[Tuple[str, int, int, str]]  # (op, i1, i2, text)

def compute_word_diff(
    line_a_tokens: List[Token],
    line_b_tokens: List[Token],
) -> List[WordDiff]:
    """
    Apskaičiuoja word-level diff su char-level detalėmis.
    
    Algoritmas:
    1. Align words by text similarity + position proximity
    2. For each aligned pair, compute char-level diff
    3. Mark unaligned words as added/deleted
    """
    words_a = [(t.text, t.bbox) for t in line_a_tokens]
    words_b = [(t.text, t.bbox) for t in line_b_tokens]
    
    result = []
    
    # Use SequenceMatcher for word sequence alignment
    sm = SequenceMatcher(None, 
                         [w[0].lower() for w in words_a],
                         [w[0].lower() for w in words_b])
    
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k, (wa, wbb) in enumerate(zip(words_a[i1:i2], words_b[j1:j2])):
                result.append(WordDiff(
                    word_a=wa[0], word_b=wbb[0],
                    bbox_a=wa[1], bbox_b=wbb[1],
                    change_type="same",
                    char_diffs=[]
                ))
        elif tag == "replace":
            # Modified words - compute char-level diff
            for wa, wbb in zip(words_a[i1:i2], words_b[j1:j2]):
                char_sm = SequenceMatcher(None, wa[0], wbb[0])
                char_diffs = [
                    (op, ci1, ci2, wa[0][ci1:ci2] if op != "insert" else wbb[0][cj1:cj2])
                    for op, ci1, ci2, cj1, cj2 in char_sm.get_opcodes()
                    if op != "equal"
                ]
                result.append(WordDiff(
                    word_a=wa[0], word_b=wbb[0],
                    bbox_a=wa[1], bbox_b=wbb[1],
                    change_type="modified",
                    char_diffs=char_diffs
                ))
        elif tag == "delete":
            for wa in words_a[i1:i2]:
                result.append(WordDiff(
                    word_a=wa[0], word_b="",
                    bbox_a=wa[1], bbox_b=None,
                    change_type="deleted",
                    char_diffs=[]
                ))
        elif tag == "insert":
            for wbb in words_b[j1:j2]:
                result.append(WordDiff(
                    word_a="", word_b=wbb[0],
                    bbox_a=None, bbox_b=wbb[1],
                    change_type="added",
                    char_diffs=[]
                ))
    
    return result
```

### 3.2 Table Diff (Cell-Level + Borders)

**Esamas sprendimas:** `comparison/table_comparison.py::compare_tables()`  
**Problema:** Coarse table comparison, nėra cell-level.

**Naujas modulis:** `comparison/cell_by_cell_table.py` (jau egzistuoja skeleton)

```python
# comparison/cell_by_cell_table.py - pilna implementacija
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from rapidfuzz import fuzz

@dataclass
class TableCell:
    row: int
    col: int
    text: str
    bbox: dict
    rowspan: int = 1
    colspan: int = 1
    style: Optional[dict] = None

@dataclass 
class CellDiff:
    row: int
    col: int
    cell_a: Optional[TableCell]
    cell_b: Optional[TableCell]
    change_type: str  # "same" | "modified" | "added" | "deleted"
    text_similarity: float
    style_changed: bool

def extract_table_cells(
    page_image: np.ndarray,
    table_bbox: dict,
    lines: List,
) -> List[TableCell]:
    """
    Išgauti cells iš table regiono.
    
    Algoritmas:
    1. Crop table region
    2. Detect horizontal/vertical lines (Hough transform)
    3. Find grid intersections
    4. Assign text lines to cells by bbox overlap
    """
    # Y-clustering for rows
    y_coords = sorted(set(
        line.bbox["y"] for line in lines 
        if _bbox_overlap(line.bbox, table_bbox) > 0.5
    ))
    row_boundaries = _cluster_1d(y_coords, threshold=0.02)
    
    # X-clustering for columns
    x_coords = sorted(set(
        line.bbox["x"] for line in lines
        if _bbox_overlap(line.bbox, table_bbox) > 0.5
    ))
    col_boundaries = _cluster_1d(x_coords, threshold=0.02)
    
    cells = []
    for row_idx, (y_min, y_max) in enumerate(zip(row_boundaries[:-1], row_boundaries[1:])):
        for col_idx, (x_min, x_max) in enumerate(zip(col_boundaries[:-1], col_boundaries[1:])):
            cell_bbox = {
                "x": x_min, "y": y_min,
                "width": x_max - x_min, "height": y_max - y_min
            }
            # Collect text in this cell
            cell_text = " ".join(
                line.text for line in lines
                if _bbox_overlap(line.bbox, cell_bbox) > 0.7
            )
            cells.append(TableCell(
                row=row_idx, col=col_idx,
                text=cell_text.strip(),
                bbox=cell_bbox
            ))
    
    return cells

def compare_tables_cell_level(
    cells_a: List[TableCell],
    cells_b: List[TableCell],
    text_threshold: float = 0.85,
) -> List[CellDiff]:
    """
    Cell-by-cell table comparison.
    
    Algoritmas:
    1. Build grid from both tables
    2. Align by (row, col) position
    3. Compare text content with fuzzy matching
    4. Detect row/column insertions by shift patterns
    """
    # Build grid lookup
    grid_a = {(c.row, c.col): c for c in cells_a}
    grid_b = {(c.row, c.col): c for c in cells_b}
    
    all_positions = set(grid_a.keys()) | set(grid_b.keys())
    diffs = []
    
    for pos in sorted(all_positions):
        cell_a = grid_a.get(pos)
        cell_b = grid_b.get(pos)
        
        if cell_a and cell_b:
            # Both exist - compare
            similarity = fuzz.ratio(
                cell_a.text.lower(), 
                cell_b.text.lower()
            ) / 100.0
            
            if similarity >= text_threshold:
                change_type = "same"
            else:
                change_type = "modified"
            
            diffs.append(CellDiff(
                row=pos[0], col=pos[1],
                cell_a=cell_a, cell_b=cell_b,
                change_type=change_type,
                text_similarity=similarity,
                style_changed=_style_differs(cell_a.style, cell_b.style)
            ))
        elif cell_a and not cell_b:
            diffs.append(CellDiff(
                row=pos[0], col=pos[1],
                cell_a=cell_a, cell_b=None,
                change_type="deleted",
                text_similarity=0.0,
                style_changed=False
            ))
        else:
            diffs.append(CellDiff(
                row=pos[0], col=pos[1],
                cell_a=None, cell_b=cell_b,
                change_type="added",
                text_similarity=0.0,
                style_changed=False
            ))
    
    return diffs

def detect_border_changes(
    page_image_a: np.ndarray,
    page_image_b: np.ndarray,
    table_bbox: dict,
) -> dict:
    """
    Aptikti table border pakeitimus naudojant drawing signature.
    
    Algoritmas:
    1. Crop table region from both images
    2. Edge detection (Canny)
    3. Hough line detection
    4. Compare line counts and positions
    """
    import cv2
    
    def _get_table_lines(img, bbox):
        # Crop
        h, w = img.shape[:2]
        x1 = int(bbox["x"] * w)
        y1 = int(bbox["y"] * h)
        x2 = int((bbox["x"] + bbox["width"]) * w)
        y2 = int((bbox["y"] + bbox["height"]) * h)
        crop = img[y1:y2, x1:x2]
        
        # Edge detection
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=20, maxLineGap=5)
        
        h_lines = []
        v_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < abs(x2 - x1):  # Horizontal
                    h_lines.append((y1 + y2) / 2)
                else:  # Vertical
                    v_lines.append((x1 + x2) / 2)
        
        return {"horizontal": len(h_lines), "vertical": len(v_lines)}
    
    lines_a = _get_table_lines(page_image_a, table_bbox)
    lines_b = _get_table_lines(page_image_b, table_bbox)
    
    return {
        "horizontal_diff": lines_b["horizontal"] - lines_a["horizontal"],
        "vertical_diff": lines_b["vertical"] - lines_a["vertical"],
        "border_changed": (
            abs(lines_b["horizontal"] - lines_a["horizontal"]) > 1 or
            abs(lines_b["vertical"] - lines_a["vertical"]) > 1
        )
    }
```

### 3.3 Figure/Image Diff (Perceptual Hash + Caption)

**Esamas sprendimas:** `comparison/figure_comparison.py::compare_figure_captions()`  
**Problema:** Tik caption comparison, nėra image comparison.

**Patobulinimas:**
```python
# comparison/figure_comparison.py - pridėti perceptual hash
import imagehash
from PIL import Image
import io

def compare_figures(
    figure_region_a: dict,  # {"bbox": {...}, "image_bytes": bytes}
    figure_region_b: dict,
    caption_a: str,
    caption_b: str,
    *,
    hash_threshold: int = 8,  # Hamming distance threshold
    caption_threshold: float = 0.85,
) -> dict:
    """
    Compare figures using perceptual hash + caption similarity.
    
    Returns:
        {
            "image_similarity": float (0-1),
            "caption_similarity": float (0-1),
            "is_same_figure": bool,
            "hash_distance": int,
        }
    """
    # Perceptual hash comparison
    try:
        img_a = Image.open(io.BytesIO(figure_region_a["image_bytes"]))
        img_b = Image.open(io.BytesIO(figure_region_b["image_bytes"]))
        
        # Use multiple hash types for robustness
        phash_a = imagehash.phash(img_a)
        phash_b = imagehash.phash(img_b)
        dhash_a = imagehash.dhash(img_a)
        dhash_b = imagehash.dhash(img_b)
        
        phash_dist = phash_a - phash_b
        dhash_dist = dhash_a - dhash_b
        
        # Combined distance (average)
        hash_distance = (phash_dist + dhash_dist) / 2
        image_similarity = max(0.0, 1.0 - hash_distance / 64.0)
    except Exception:
        hash_distance = 64
        image_similarity = 0.0
    
    # Caption comparison
    from rapidfuzz import fuzz
    caption_similarity = fuzz.ratio(
        caption_a.lower() if caption_a else "",
        caption_b.lower() if caption_b else ""
    ) / 100.0
    
    is_same = (
        hash_distance <= hash_threshold and
        caption_similarity >= caption_threshold
    )
    
    return {
        "image_similarity": image_similarity,
        "caption_similarity": caption_similarity,
        "is_same_figure": is_same,
        "hash_distance": hash_distance,
    }
```

### 3.4 Formula Diff (LaTeX OCR)

**Naujas modulis:** `comparison/formula_comparison.py`

```python
# comparison/formula_comparison.py
from dataclasses import dataclass
from typing import Optional
from rapidfuzz import fuzz

@dataclass
class FormulaRegion:
    bbox: dict
    latex: Optional[str] = None  # OCR result
    image_bytes: Optional[bytes] = None
    confidence: float = 0.0

def extract_formula_latex(
    image_bytes: bytes,
    engine: str = "pix2tex",
) -> tuple[str, float]:
    """
    Extract LaTeX from formula image using OCR.
    
    Engines:
    - pix2tex: LaTeX-OCR neural network
    - mathpix: Commercial API (more accurate)
    - tesseract: Fallback (poor for math)
    """
    if engine == "pix2tex":
        try:
            from pix2tex.cli import LatexOCR
            model = LatexOCR()
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(image_bytes))
            latex = model(img)
            return latex, 0.85  # pix2tex doesn't give confidence
        except ImportError:
            pass
    
    # Fallback: treat as text
    return "", 0.0

def compare_formulas(
    formula_a: FormulaRegion,
    formula_b: FormulaRegion,
    *,
    latex_threshold: float = 0.90,
) -> dict:
    """
    Compare two formulas by LaTeX similarity.
    
    Normalization:
    - Remove whitespace
    - Normalize common variants (\\frac vs \\dfrac)
    - Compare structure ignoring variable names optionally
    """
    def normalize_latex(latex: str) -> str:
        if not latex:
            return ""
        # Remove whitespace
        latex = "".join(latex.split())
        # Normalize common variants
        latex = latex.replace("\\dfrac", "\\frac")
        latex = latex.replace("\\left(", "(")
        latex = latex.replace("\\right)", ")")
        latex = latex.replace("\\cdot", "*")
        return latex
    
    norm_a = normalize_latex(formula_a.latex or "")
    norm_b = normalize_latex(formula_b.latex or "")
    
    if not norm_a and not norm_b:
        # Both empty - use visual comparison
        from comparison.figure_comparison import compare_figures
        visual = compare_figures(
            {"image_bytes": formula_a.image_bytes, "bbox": formula_a.bbox},
            {"image_bytes": formula_b.image_bytes, "bbox": formula_b.bbox},
            "", "",
            hash_threshold=5,  # Stricter for formulas
        )
        return {
            "latex_similarity": 0.0,
            "visual_similarity": visual["image_similarity"],
            "is_same_formula": visual["image_similarity"] > 0.9,
            "method": "visual",
        }
    
    similarity = fuzz.ratio(norm_a, norm_b) / 100.0
    
    return {
        "latex_similarity": similarity,
        "visual_similarity": None,
        "is_same_formula": similarity >= latex_threshold,
        "method": "latex",
        "latex_a": norm_a,
        "latex_b": norm_b,
    }
```

### 3.5 Layout Diff (Reflow vs Move)

**Esamas sprendimas:** `comparison/alignment.py::detect_layout_shift()`  
**Problema:** Paprastas Y-shift detection, neatskiria reflow nuo move.

**Patobulinimas:**
```python
# comparison/alignment.py - papildyti
from enum import Enum

class LayoutChangeType(Enum):
    NONE = "none"
    REFLOW = "reflow"      # Text reflowed (line breaks changed)
    MOVE = "move"          # Block moved to different position
    RESIZE = "resize"      # Block size changed
    REORDER = "reorder"    # Reading order changed

def classify_layout_change(
    block_a: TextBlock,
    block_b: TextBlock,
    *,
    position_tolerance: float = 0.05,  # 5% of page
    size_tolerance: float = 0.1,       # 10% size change
) -> LayoutChangeType:
    """
    Klasifikuoti layout pakeitimo tipą.
    
    Heuristikos:
    - REFLOW: Same text, different line count or line breaks
    - MOVE: Same bbox size, different position (>tolerance)
    - RESIZE: Same position, different size
    - REORDER: Same text & position, different reading_order
    """
    bbox_a = block_a.bbox
    bbox_b = block_b.bbox
    
    # Position difference
    dx = abs(bbox_a["x"] - bbox_b["x"])
    dy = abs(bbox_a["y"] - bbox_b["y"])
    
    # Size difference
    dw = abs(bbox_a["width"] - bbox_b["width"]) / max(bbox_a["width"], 0.01)
    dh = abs(bbox_a["height"] - bbox_b["height"]) / max(bbox_a["height"], 0.01)
    
    position_changed = dx > position_tolerance or dy > position_tolerance
    size_changed = dw > size_tolerance or dh > size_tolerance
    
    # Check for reflow by line count
    lines_a = len(block_a.lines) if hasattr(block_a, 'lines') else 1
    lines_b = len(block_b.lines) if hasattr(block_b, 'lines') else 1
    line_count_changed = lines_a != lines_b
    
    if line_count_changed and not position_changed:
        return LayoutChangeType.REFLOW
    elif position_changed and not size_changed:
        return LayoutChangeType.MOVE
    elif size_changed and not position_changed:
        return LayoutChangeType.RESIZE
    elif not position_changed and not size_changed:
        return LayoutChangeType.NONE
    else:
        # Both position and size changed - likely move+reflow
        return LayoutChangeType.MOVE
```

---

## 4. Output Schema (C sritis)

### 4.1 Unified Diff Schema

```python
# comparison/models.py - atnaujinti Diff klasę
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from enum import Enum

class DiffType(Enum):
    ADDED = "added"
    DELETED = "deleted"
    MODIFIED = "modified"

class ChangeCategory(Enum):
    CONTENT = "content"       # Text changed
    FORMATTING = "formatting" # Style changed
    LAYOUT = "layout"         # Position/size changed
    VISUAL = "visual"         # Pixel-level change

@dataclass
class UnifiedDiff:
    """
    Unified diff format for all comparison results.
    """
    diff_id: str                      # Unique identifier
    page_a: int                       # Page number in doc A
    page_b: int                       # Page number in doc B (may differ)
    
    diff_type: DiffType               # added/deleted/modified
    change_category: ChangeCategory   # content/formatting/layout/visual
    
    # Location
    bbox_a: Optional[dict] = None     # Bounding box in doc A
    bbox_b: Optional[dict] = None     # Bounding box in doc B
    
    # Content
    text_a: Optional[str] = None      # Original text
    text_b: Optional[str] = None      # Changed text
    
    # Word-level details (for content changes)
    word_diffs: List[dict] = field(default_factory=list)
    
    # Style details (for formatting changes)
    style_a: Optional[dict] = None
    style_b: Optional[dict] = None
    
    # Confidence and source
    confidence: float = 1.0
    sources: List[str] = field(default_factory=list)  # ["text_diff", "visual_diff"]
    
    # Additional metadata
    element_type: str = "text"        # text/table/figure/formula/header
    metadata: dict = field(default_factory=dict)

@dataclass
class ComparisonReport:
    """Full comparison report."""
    doc_a_path: str
    doc_b_path: str
    timestamp: str
    
    # Summary
    total_pages_a: int
    total_pages_b: int
    pages_compared: int
    
    # Diff counts by category
    content_changes: int
    formatting_changes: int
    layout_changes: int
    visual_changes: int
    
    # All diffs
    diffs: List[UnifiedDiff]
    
    # Debug info (when enabled)
    debug: Optional[dict] = None
    
    # Metrics
    comparison_time_seconds: float = 0.0
    extraction_method: str = "auto"
```

### 4.2 Debug Mode JSON Export

```python
# export/json_exporter.py - papildyti debug mode
import json
from pathlib import Path
from datetime import datetime

def export_debug_json(
    result: ComparisonReport,
    output_path: Path,
    *,
    include_raw_extraction: bool = True,
    include_intermediate_diffs: bool = True,
) -> Path:
    """
    Export full debug information as JSON.
    
    Structure:
    {
        "metadata": {...},
        "extraction_a": {...},  # Raw extraction from doc A
        "extraction_b": {...},  # Raw extraction from doc B
        "alignment": {...},     # Page/section alignment
        "raw_diffs": {
            "text": [...],
            "formatting": [...],
            "table": [...],
            "figure": [...],
            "visual": [...]
        },
        "fused_diffs": [...],   # After deduplication
        "final_diffs": [...],   # After classification
    }
    """
    debug_data = {
        "metadata": {
            "doc_a": result.doc_a_path,
            "doc_b": result.doc_b_path,
            "timestamp": result.timestamp,
            "version": "1.0.0",
        },
        "summary": {
            "pages_a": result.total_pages_a,
            "pages_b": result.total_pages_b,
            "content_changes": result.content_changes,
            "formatting_changes": result.formatting_changes,
            "layout_changes": result.layout_changes,
            "visual_changes": result.visual_changes,
        },
        "diffs": [_diff_to_dict(d) for d in result.diffs],
    }
    
    if result.debug:
        debug_data["debug"] = result.debug
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, indent=2, ensure_ascii=False)
    
    return output_path

def _diff_to_dict(diff: UnifiedDiff) -> dict:
    return {
        "id": diff.diff_id,
        "page_a": diff.page_a,
        "page_b": diff.page_b,
        "type": diff.diff_type.value,
        "category": diff.change_category.value,
        "bbox_a": diff.bbox_a,
        "bbox_b": diff.bbox_b,
        "text_a": diff.text_a,
        "text_b": diff.text_b,
        "confidence": diff.confidence,
        "sources": diff.sources,
        "element_type": diff.element_type,
    }
```

---

## 5. Patch Plan (D sritis)

### 5.1 Failai Kuriuos Reikia Modifikuoti

| Failas | Pakeitimai | Prioritetas |
|--------|-----------|-------------|
| `utils/text_normalization.py` | Pridėti `normalize_text_full()` su NBSP, zero-width | P0 |
| `extraction/line_extractor.py` | Pridėti style info į Token | P0 |
| `extraction/__init__.py` | Pridėti `_text_quality_score()` | P1 |
| `comparison/models.py` | Atnaujinti `Diff` → `UnifiedDiff` | P0 |
| `comparison/line_comparison.py` | Pridėti `compute_word_diff()` | P0 |
| `comparison/cell_by_cell_table.py` | Implementuoti cell-level comparison | P0 |
| `comparison/figure_comparison.py` | Pridėti perceptual hash | P1 |
| `comparison/alignment.py` | Pridėti `classify_layout_change()` | P1 |
| `comparison/diff_fusion.py` | Atnaujinti fusion logic su naujais typais | P1 |
| `pipeline/compare_pdfs.py` | Integruoti naujas funkcijas | P0 |
| `config/settings.py` | Nauji threshold parametrai | P0 |
| `export/json_exporter.py` | Debug mode export | P2 |

### 5.2 Nauji Failai

| Failas | Aprašymas |
|--------|-----------|
| `comparison/formula_comparison.py` | LaTeX OCR + formula diff |
| `comparison/word_diff.py` | Word-level diff su char details |
| `utils/perceptual_hash.py` | Image hashing utilities |

### 5.3 Konkretūs Patches

#### Patch 1: Text Normalization (P0)

```python
# utils/text_normalization.py - APPEND to file

def normalize_text_full(text: str, *, preserve_case: bool = False) -> str:
    """Full normalization for academic comparison."""
    if not text:
        return ""
    
    import unicodedata
    
    # NFC normalization
    text = unicodedata.normalize("NFC", text)
    
    # Special spaces → regular space
    for char in ["\u00A0", "\u202F", "\u2007", "\u2009"]:
        text = text.replace(char, " ")
    
    # Zero-width removal
    for char in ["\u200B", "\u200C", "\u200D", "\uFEFF", "\u00AD"]:
        text = text.replace(char, "")
    
    # Whitespace collapse
    text = " ".join(text.split())
    
    if not preserve_case:
        text = text.lower()
    
    return text.strip()
```

#### Patch 2: Config Settings (P0)

```python
# config/settings.py - ADD these fields to Settings class

# Cell-level table comparison
table_cell_text_threshold: float = 0.85
table_border_detection_enabled: bool = True

# Figure comparison
figure_hash_threshold: int = 8  # Hamming distance
figure_caption_threshold: float = 0.80

# Formula comparison
formula_latex_threshold: float = 0.90
formula_ocr_engine: str = "pix2tex"

# Layout classification
layout_position_tolerance: float = 0.05  # 5% of page
layout_size_tolerance: float = 0.10      # 10% change

# Debug mode
debug_mode: bool = False
debug_output_path: str = "./debug_output"
```

---

## 6. Threshold Recommendations (E sritis)

### 6.1 Rekomenduojamos Reikšmės

| Parametras | Reikšmė | Pagrindimas |
|------------|---------|-------------|
| `text_similarity_threshold` | 0.85 | Aukštesnis nei 0.82, sumažina false positives |
| `table_cell_text_threshold` | 0.85 | Cell tekstas turėtų būti labai panašus |
| `figure_hash_threshold` | 8 | Hamming distance ≤8 = ta pati nuotrauka |
| `layout_position_tolerance` | 0.05 | 5% puslapio = ~30pt ties 612pt width |
| `ocr_min_text_similarity_for_match` | 0.75 | Šiek tiek aukštesnis nei 0.7 |
| `font_size_change_threshold_pt` | 0.5 | 0.5pt = tikras dydžio pokytis |

### 6.2 Tuning Gidas

1. **Jei per daug false positives:**
   - Padidinti `text_similarity_threshold` (0.85 → 0.90)
   - Padidinti `layout_position_tolerance` (0.05 → 0.08)
   - Įjungti agresyvesnį diff fusion

2. **Jei praleidžia tikrus pakeitimus:**
   - Sumažinti `text_similarity_threshold` (0.85 → 0.80)
   - Sumažinti `figure_hash_threshold` (8 → 5)

3. **Lietuviškų raidžių problemos:**
   - Patikrinti ar NFC normalizacija veikia
   - Tikrinti ar OCR engine palaiko LT

---

## 7. Implementacijos Tvarka

### Fazė 1: Core Normalization (1-2 dienos)
- [ ] `normalize_text_full()` implementacija
- [ ] Unit testai su lietuviškomis raidėmis
- [ ] Integracija į esamus modules

### Fazė 2: Word-Level Diff (2-3 dienos)
- [ ] `compute_word_diff()` implementacija
- [ ] `WordDiff` dataclass
- [ ] Integracija į `line_comparison.py`

### Fazė 3: Cell-Level Table (3-4 dienos)
- [ ] `extract_table_cells()` su clustering
- [ ] `compare_tables_cell_level()`
- [ ] Border detection su OpenCV

### Fazė 4: Figure/Formula (2-3 dienos)
- [ ] Perceptual hash integration
- [ ] Optional: pix2tex for LaTeX OCR
- [ ] Visual fallback

### Fazė 5: Integration & Testing (2-3 dienos)
- [ ] Pipeline integration
- [ ] Config updates
- [ ] End-to-end testing
- [ ] Debug JSON export

---

## 8. Testing Strategy

### Unit Tests

```python
# tests/test_normalization.py
def test_lithuanian_nfc():
    """Lietuviškos raidės turi būti išsaugotos po normalizacijos."""
    text = "Ąžuolas ėjo į mokyklą"
    normalized = normalize_text_full(text, preserve_case=True)
    assert "Ąžuolas" in normalized
    assert "į" in normalized

def test_nbsp_removal():
    """NBSP turi virsti paprastu tarpu."""
    text = "Hello\u00A0World"
    normalized = normalize_text_full(text)
    assert "\u00A0" not in normalized
    assert "hello world" == normalized
```

### Integration Tests

```python
# tests/test_cell_level_table.py
def test_table_cell_change():
    """Aptikti vienos celės pakeitimą."""
    cells_a = [TableCell(0, 0, "Alpha", {}), TableCell(0, 1, "Beta", {})]
    cells_b = [TableCell(0, 0, "Alpha", {}), TableCell(0, 1, "Gamma", {})]
    
    diffs = compare_tables_cell_level(cells_a, cells_b)
    
    modified = [d for d in diffs if d.change_type == "modified"]
    assert len(modified) == 1
    assert modified[0].col == 1
```

---

## 9. Rizikos ir Mitigacija

| Rizika | Tikimybė | Poveikis | Mitigacija |
|--------|----------|----------|------------|
| OCR kokybė lietuviškam tekstui | Vidutinė | Aukštas | Naudoti PaddleOCR su LT modeliu |
| Table detection neaptinka sudėtingų tables | Aukšta | Vidutinis | Fallback į visual diff |
| pix2tex neprieinamas | Žema | Žemas | Visual hash fallback |
| Performance su dideliais PDF | Vidutinė | Vidutinis | Lazy loading, caching |

---

*Dokumentas paruoštas: 2025-01*
