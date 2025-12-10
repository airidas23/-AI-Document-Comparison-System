"""
Layout detection for tables, figures, paragraphs, headers, signatures, and more.

Replaces Detectron2 with YOLOv11 + SAM (Apple Silicon compatible).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import List

from comparison.models import PageData
from config.settings import settings
from utils.logging import logger

# ---------------------------------------------------------
# DocLayout-YOLO + SAM IMPORTS
# ---------------------------------------------------------
try:
    # DocLayout-YOLO uses YOLOv10 architecture for document layout detection
    from doclayout_yolo import YOLOv10 as DocLayoutYOLO

    DOCLAYOUT_YOLO_AVAILABLE = True
except Exception:
    DOCLAYOUT_YOLO_AVAILABLE = False

try:
    # Fallback to standard ultralytics YOLO for other models
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

try:
    from segment_anything import sam_model_registry, SamPredictor

    SAM_AVAILABLE = True
except Exception:
    SAM_AVAILABLE = False

# Lazy loaders for models
_yolo_model = None
_sam_predictor = None
_is_doclayout_model = False  # Track which model type is loaded


def load_yolo_model():
    """Load DocLayout-YOLO or standard YOLO model once."""
    global _yolo_model, _is_doclayout_model
    
    if _yolo_model is not None:
        return _yolo_model
        
    model_name = settings.yolo_layout_model_name
    
    # Convert to absolute path if it's a relative path
    model_path = Path(model_name)
    if not model_path.is_absolute():
        # Assume relative to project root (parent of config dir)
        project_root = Path(__file__).parent.parent
        model_path = project_root / model_name
    
    # Check if model file exists
    if not model_path.exists():
        logger.warning(
            "YOLO model file not found at: %s (resolved from: %s).",
            model_path, model_name
        )
        return None
    
    # Determine if this is a DocLayout-YOLO model based on filename
    is_doclayout = "doclayout" in model_name.lower()
    
    if is_doclayout and DOCLAYOUT_YOLO_AVAILABLE:
        try:
            # Use DocLayout-YOLO for document-specific models
            device = "cuda" if settings.use_gpu else "cpu"
            _yolo_model = DocLayoutYOLO(str(model_path)).to(device)
            _is_doclayout_model = True
            logger.info("DocLayout-YOLO model loaded successfully: %s (device: %s)", model_path, device)
        except Exception as exc:
            logger.warning(
                "Failed to load DocLayout-YOLO (path: %s): %s", 
                model_path, exc
            )
            _yolo_model = None
    elif YOLO_AVAILABLE:
        try:
            # Use standard ultralytics YOLO for other models
            _yolo_model = YOLO(str(model_path))
            _is_doclayout_model = False
            logger.info("Standard YOLO model loaded successfully: %s", model_path)
        except Exception as exc:
            logger.warning(
                "Failed to load YOLO (path: %s): %s", 
                model_path, exc
            )
            _yolo_model = None
    else:
        logger.warning("No YOLO library available for model: %s", model_path)
        _yolo_model = None
        
    return _yolo_model


def load_sam_model():
    """Load SAM model only if segmentation refinement is needed."""
    global _sam_predictor
    if _sam_predictor is None and SAM_AVAILABLE:
        try:
            checkpoint = settings.sam_checkpoint_path or "sam_vit_h_4b8939.pth"
            sam = sam_model_registry["vit_h"](checkpoint)
            _sam_predictor = SamPredictor(sam)
            logger.info("SAM model loaded successfully: %s", checkpoint)
        except Exception as exc:
            logger.warning("Failed to load SAM (%s): %s", settings.sam_checkpoint_path, exc)
            _sam_predictor = None
    return _sam_predictor


# ---------------------------------------------------------
# YOLO → Domain Label Mapping
# Updated for DocLayout-YOLO document-specific classes
# Model classes: title, plain text, abandon, figure, figure_caption,
#                table, table_caption, table_footnote, isolate_formula, formula_caption
# ---------------------------------------------------------
YOLO_LAYOUT_MAP = {
    # DocLayout-YOLO actual model classes
    "title": "title",
    "plain text": "paragraph",
    "abandon": "other",  # Content to be abandoned/ignored
    "figure": "figure",
    "figure_caption": "caption",
    "table": "table",
    "table_caption": "caption",
    "table_footnote": "footer",
    "isolate_formula": "formula",
    "formula_caption": "caption",
    
    # Legacy/alternative mappings for compatibility
    "caption": "caption",
    "footnote": "footer",
    "formula": "formula",
    "list-item": "list",
    "page-footer": "footer",
    "page-header": "header",
    "picture": "figure",
    "section-header": "title",
    "text": "paragraph",
    "textblock": "paragraph",
    "paragraph": "paragraph",
    "header": "header",
    "footer": "footer",
    "signature": "signature",
    "stamp": "signature",
    "logo": "figure",
    "image": "figure",
}


# ---------------------------------------------------------
# MAIN ENTRY
# ---------------------------------------------------------
def analyze_layout(path: str | Path, use_layoutparser: bool = True) -> List[PageData]:
    """
    Document layout analysis using DocLayout-YOLO, standard YOLO, or heuristic fallback.
    DocLayout-YOLO is preferred for document-specific layout detection.
    """
    import fitz  # PyMuPDF

    path = Path(path)
    logger.info("Analyzing layout for: %s", path)

    # Prefer DocLayout-YOLO, fall back to standard YOLO, then heuristics
    use_yolo = DOCLAYOUT_YOLO_AVAILABLE or YOLO_AVAILABLE
    use_sam = SAM_AVAILABLE

    if use_yolo:
        layout_mode_reason = "doclayout_yolo" if DOCLAYOUT_YOLO_AVAILABLE else "yolo"
    else:
        layout_mode_reason = "heuristic"

    doc = fitz.open(path)
    pages: List[PageData] = []

    for page in doc:
        page_data = PageData(
            page_num=page.number + 1,
            width=page.rect.width,
            height=page.rect.height,
        )

        if use_yolo:
            tables, figures, text_regions = _detect_with_yolo(page)
        else:
            tables = _detect_tables(page)
            figures = _detect_figures(page)
            text_regions = []

        page_data.metadata.update({
            "tables": tables,
            "figures": figures,
            "text_blocks": text_regions,
            "layout_analyzed": True,
            "layout_method": layout_mode_reason,
        })

        pages.append(page_data)

    doc.close()
    logger.info("Layout analysis complete: %d pages", len(pages))
    return pages


# ---------------------------------------------------------
# DOCASLAY-YOLO / YOLO DETECTION
# ---------------------------------------------------------
def _detect_with_yolo(page) -> tuple[List[dict], List[dict], List[dict]]:
    from PIL import Image
    import numpy as np

    model = load_yolo_model()
    if model is None:
        return [], [], []

    # Convert PDF page → image
    pix = page.get_pixmap(dpi=150)
    img = Image.open(io.BytesIO(pix.tobytes("png")))

    conf_th = getattr(settings, "yolo_layout_confidence", 0.3)
    iou_th = 0.45  # Default IoU threshold

    # Use different prediction methods based on model type
    if _is_doclayout_model:
        # DocLayout-YOLO uses predict() method with specific parameters
        # imgsz=1024 matches the model name: doclayout_yolo_docstructbench_imgsz1024.pt
        prediction = model.predict(
            img,
            imgsz=1024,
            conf=conf_th,
            iou=iou_th,
            verbose=False
        )[0]
    else:
        # Standard ultralytics YOLO uses __call__ or predict
        prediction = model(img, conf=conf_th)[0]

    tables = []
    figures = []
    paragraphs = []

    # Parse the prediction results
    # DocLayout-YOLO returns prediction with boxes attribute
    if not hasattr(prediction, "boxes") or prediction.boxes is None:
        return tables, figures, paragraphs

    for xyxy, conf, cls in zip(
        prediction.boxes.xyxy.cpu(),
        prediction.boxes.conf.cpu(),
        prediction.boxes.cls.cpu(),
    ):
        cls_idx = int(cls.item())
        cls_name = model.names[cls_idx].lower()
        bbox = xyxy.tolist()
        confidence = float(conf.item())

        # Map YOLO class → our domain layout class
        mapped = YOLO_LAYOUT_MAP.get(cls_name, None)
        if mapped is None:
            continue

        entry = {
            "bbox": bbox,
            "confidence": round(confidence, 3),
            "label": mapped,
        }

        if mapped == "table":
            tables.append(entry)
        elif mapped == "figure":
            figures.append(entry)
        elif mapped in ("paragraph", "title", "header", "footer"):
            paragraphs.append(entry)

    return tables, figures, paragraphs


# ---------------------------------------------------------
# OPTIONAL — SAM MASK REFINEMENT
# ---------------------------------------------------------
def refine_with_sam(img, bbox):
    """Generate a segmentation mask for a detected region."""
    predictor = load_sam_model()
    if predictor is None:
        return None

    import numpy as np

    x0, y0, x1, y1 = map(int, bbox)
    predictor.set_image(np.array(img))
    input_box = np.array([x0, y0, x1, y1])

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    return masks[0].astype(np.uint8).tolist()


# ---------------------------------------------------------
# HEURISTIC FALLBACKS (FROM YOUR ORIGINAL FILE)
# ---------------------------------------------------------
def _detect_tables(page) -> List[dict]:
    """Fallback heuristic table detection."""
    tables = []
    try:
        blocks = page.get_text("dict", flags=0)
        for block in blocks.get("blocks", []):
            if block.get("type") == 0:
                lines = block.get("lines", [])
                if len(lines) > 3:
                    x_coords = []
                    for line in lines:
                        for span in line.get("spans", []):
                            bbox = span.get("bbox", [])
                            if len(bbox) == 4:
                                x_coords.append(bbox[0])

                    if len(set(round(x, 0) for x in x_coords)) < len(x_coords) * 0.7:
                        tables.append({
                            "bbox": block.get("bbox"),
                            "confidence": 0.5,
                        })
    except Exception as exc:
        logger.debug("Table detection error: %s", exc)

    return tables


def _detect_figures(page) -> List[dict]:
    """Fallback heuristic for image detection."""
    figures = []
    try:
        image_list = page.get_images()
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            img_rects = page.get_image_rects(xref)
            for rect in img_rects:
                figures.append({
                    "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
                    "xref": xref,
                    "width": base_image["width"],
                    "height": base_image["height"],
                    "confidence": 1.0,
                })
    except Exception as exc:
        logger.debug("Figure detection error: %s", exc)

    return figures

