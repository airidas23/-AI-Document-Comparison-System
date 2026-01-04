"""Gradio interface components."""
from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import numpy as np

from comparison.diff_classifier import classify_diffs, get_diff_summary
from comparison.figure_comparison import compare_figure_captions
from comparison.formatting_comparison import compare_formatting
from comparison.models import ComparisonResult
from comparison.table_comparison import compare_tables
from comparison.text_comparison import TextComparator
from config.settings import settings
from export.json_exporter import export_json
from export.pdf_exporter import export_pdf
from extraction import extract_pdf
from extraction.header_footer_detector import compare_headers_footers
from extraction.ocr_router import is_cuda_available
from utils.logging import logger
from utils.performance import clear_timings, track_time
from utils.validation import validate_pdf_path
from visualization.pdf_viewer import render_pages


GLOBAL_CLIENT_LOAD_JS = r"""
// =====================================================
// Global client JS for the Gradio app (loaded via demo.load(..., js=...))
// - Ensures scroll sync works even when <script> inside gr.HTML updates doesn't execute.
// - Provides window.pdfViewerA/B + window.highlightDiffOnViewers for jump/highlight behavior.
// =====================================================
() => {
  if (window.__SYNC_VIEWER_GLOBAL_INIT__) return;
  window.__SYNC_VIEWER_GLOBAL_INIT__ = true;

  const LOG_PREFIX = "[sync-viewer]";

  const DIFF_COLORS = {
    added: { border: "rgba(0, 255, 0, 0.7)", background: "rgba(0, 255, 0, 0.2)", glow: "rgba(0, 255, 0, 0.5)" },
    deleted: { border: "rgba(255, 0, 0, 0.7)", background: "rgba(255, 0, 0, 0.2)", glow: "rgba(255, 0, 0, 0.5)" },
    modified: { border: "rgba(255, 215, 0, 0.7)", background: "rgba(255, 215, 0, 0.2)", glow: "rgba(255, 215, 0, 0.5)" }
  };

  function getViewerEl(id) {
    return document.getElementById(`sync-img-viewer-${id}`);
  }

  function parseAlignmentFromDom(aEl, bEl) {
    const raw = (aEl && aEl.dataset && aEl.dataset.alignment) || (bEl && bEl.dataset && bEl.dataset.alignment) || "";
    if (!raw) return {};
    try {
      return JSON.parse(raw);
    } catch (e) {
      return {};
    }
  }

  function invertMap(mapObj) {
    const inv = {};
    try {
      Object.keys(mapObj || {}).forEach((k) => {
        const v = mapObj[k];
        if (v != null) inv[String(v)] = String(k);
      });
    } catch (e) {}
    return inv;
  }

  function buildViewer(id) {
    const container = getViewerEl(id);
    if (!container) return null;

    function getPageEl(pageNum) {
      return container.querySelector(`.sync-page[data-page="${pageNum}"]`);
    }

    function getActivePageNum() {
      const pages = container.querySelectorAll(".sync-page");
      if (!pages || pages.length === 0) return null;
      const y = container.scrollTop;
      const bias = 24;
      let active = pages[0];
      for (let i = 0; i < pages.length; i++) {
        const p = pages[i];
        if (p.offsetTop <= y + bias) active = p;
        else break;
      }
      const pageStr = active && active.dataset ? active.dataset.page : null;
      const n = pageStr ? Number(pageStr) : null;
      return Number.isFinite(n) ? n : null;
    }

    function scrollToPage(pageNum) {
      const pageEl = getPageEl(pageNum);
      if (!pageEl) return;
      try {
        container.scrollTo({ top: pageEl.offsetTop, behavior: "smooth" });
      } catch (e) {
        container.scrollTop = pageEl.offsetTop;
      }
    }

    function clearHighlights() {
      container.querySelectorAll(".sync-bbox-overlay").forEach((el) => el.remove());
    }

    function highlightBBox(pageNum, bbox, diffType) {
      if (!bbox) return null;
      const pageEl = getPageEl(pageNum);
      if (!pageEl) return null;
      const inner = pageEl.querySelector(".sync-page-inner");
      if (!inner) return null;

      const colors = DIFF_COLORS[diffType || "modified"] || DIFF_COLORS.modified;
      const x = Number(bbox.x ?? bbox.left ?? 0);
      const y = Number(bbox.y ?? bbox.top ?? 0);
      const w = Number(bbox.width ?? bbox.w ?? 0);
      const h = Number(bbox.height ?? bbox.h ?? 0);

      const overlay = document.createElement("div");
      overlay.className = "sync-bbox-overlay";

      const iw = inner.clientWidth || inner.getBoundingClientRect().width || 1;
      const ih = inner.clientHeight || inner.getBoundingClientRect().height || 1;
      const cx = Math.max(0, Math.min(1, x)) * iw;
      const cy = Math.max(0, Math.min(1, y)) * ih;
      const cw = Math.max(0, Math.min(1, w)) * iw;
      const ch = Math.max(0, Math.min(1, h)) * ih;

      overlay.style.left = `${cx}px`;
      overlay.style.top = `${cy}px`;
      overlay.style.width = `${cw}px`;
      overlay.style.height = `${ch}px`;
      overlay.style.borderColor = colors.border;
      overlay.style.backgroundColor = colors.background;
      overlay.style.boxShadow = `0 0 10px ${colors.glow}`;

      inner.appendChild(overlay);
      return overlay;
    }

    return { id, container, getActivePageNum, scrollToPage, clearHighlights, highlightBBox };
  }

  const state = {
    isSyncing: false,
    lastPage: { a: null, b: null },
    timers: { a: null, b: null }
  };

  function bindScrollSync(viewerA, viewerB) {
    if (!viewerA || !viewerB) return;

    // Expose for jump/highlight helpers.
    window.pdfViewerA = viewerA;
    window.pdfViewerB = viewerB;

    const alignment = parseAlignmentFromDom(viewerA.container, viewerB.container);
    const pageMapAtoB = (alignment && (alignment.page_map || alignment.alignment_map)) || {};
    const pageMapBtoA = invertMap(pageMapAtoB);

    function mapPage(sourceId, pageNum) {
      if (!pageNum) return pageNum;
      const map = sourceId === "a" ? pageMapAtoB : pageMapBtoA;
      const mapped = map && map[String(pageNum)];
      const n = mapped != null ? Number(mapped) : null;
      return Number.isFinite(n) ? n : pageNum;
    }

    function attach(id, src, dst) {
      const el = src.container;
      if (!el || el.dataset.scrollSyncBound === "1") return;
      el.dataset.scrollSyncBound = "1";

      el.addEventListener(
        "scroll",
        () => {
          if (state.timers[id]) clearTimeout(state.timers[id]);
          state.timers[id] = setTimeout(() => {
            if (state.isSyncing) return;

            const sourcePage = src.getActivePageNum();
            if (!sourcePage) return;
            if (state.lastPage[id] === sourcePage) return;
            state.lastPage[id] = sourcePage;

            const targetPage = mapPage(id, sourcePage);
            const targetActive = dst.getActivePageNum();
            if (targetActive === targetPage) return;

            state.isSyncing = true;
            dst.scrollToPage(targetPage);
            setTimeout(() => {
              state.isSyncing = false;
            }, 200);
          }, 60);
        },
        { passive: true }
      );
    }

    attach("a", viewerA, viewerB);
    attach("b", viewerB, viewerA);
  }

  window.highlightDiffOnViewers = function (diff) {
    try {
      const a = window.pdfViewerA;
      const b = window.pdfViewerB;
      if (!a || !b || !diff) return;

      const pageNumA = diff.page_num || diff.page;
      const pageNumB = diff.page_num_b || diff.page_num || diff.page;
      const bboxA = diff.bbox;
      const bboxB = diff.bbox_b || diff.bbox;
      const diffType = diff.diff_type || "modified";
      if (!pageNumA || !bboxA) return;

      a.clearHighlights();
      b.clearHighlights();
      a.highlightBBox(pageNumA, bboxA, diffType);
      if (pageNumB && bboxB) b.highlightBBox(pageNumB, bboxB, diffType);
      a.scrollToPage(pageNumA);
      if (pageNumB) b.scrollToPage(pageNumB);
    } catch (e) {}
  };

  function initIfReady() {
    const aEl = getViewerEl("a");
    const bEl = getViewerEl("b");
    if (!aEl || !bEl) return;
    const viewerA = buildViewer("a");
    const viewerB = buildViewer("b");
    if (!viewerA || !viewerB) return;
    bindScrollSync(viewerA, viewerB);
  }

  const observer = new MutationObserver(() => initIfReady());
  observer.observe(document.body, { childList: true, subtree: true });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => setTimeout(initIfReady, 50));
  } else {
    setTimeout(initIfReady, 50);
  }
  document.addEventListener("gradio:loaded", () => setTimeout(initIfReady, 50));

  try {
    console.log(`${LOG_PREFIX} global JS loaded`);
  } catch (e) {}

  // =====================================================
  // Zoom controls (Gallery + Sync viewer)
  // - Works with Gradio 6 DOM changes by wiring by elem_id when possible,
  //   and falling back to event delegation.
  // =====================================================
  if (!window.__GLOBAL_ZOOM_STATE__) {
    window.__GLOBAL_ZOOM_STATE__ = { a: 100, b: 100 };
  }

  function clampZoomPercent(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return 100;
    return Math.max(50, Math.min(200, n));
  }

  function applyZoomToGalleryRoot(rootEl, percent) {
    if (!rootEl) return;
    const p = clampZoomPercent(percent);
    const imgs = rootEl.querySelectorAll("img");
    if (!imgs || imgs.length === 0) return;

    // Apply width scaling to images (layout-safe); avoids relying on CSS zoom.
    imgs.forEach((img) => {
      img.style.maxWidth = "none";
      img.style.width = p === 100 ? "" : `${p}%`;
    });
  }

  function applyZoomToSyncViewer(viewerId, percent) {
    const p = clampZoomPercent(percent);
    const el = document.getElementById(`sync-img-viewer-${viewerId}`);
    if (!el) return;
    const imgs = el.querySelectorAll(".sync-page-img, img");
    if (!imgs || imgs.length === 0) return;
    imgs.forEach((img) => {
      img.style.maxWidth = "none";
      img.style.width = p === 100 ? "" : `${p}%`;
    });
  }

  function applyZoom(docId, percent) {
    const p = clampZoomPercent(percent);
    window.__GLOBAL_ZOOM_STATE__[docId] = p;

    // Gallery IDs are stable in code: gallery-a / gallery-b
    applyZoomToGalleryRoot(document.getElementById(docId === "a" ? "gallery-a" : "gallery-b"), p);
    applyZoomToSyncViewer(docId, p);
  }

  function setControlValues(containerEl, percent) {
    if (!containerEl) return;
    const p = clampZoomPercent(percent);
    const range = containerEl.querySelector('input[type="range"]');
    const num = containerEl.querySelector('input[type="number"]');
    if (range) range.value = String(p);
    if (num) num.value = String(p);
  }

  function wireZoomControl(docId, sliderRootId) {
    const container = document.getElementById(sliderRootId);
    if (!container) return false;
    if (container.dataset.zoomBound === "1") return true;
    container.dataset.zoomBound = "1";

    // Keep UI in sync with current state.
    setControlValues(container, window.__GLOBAL_ZOOM_STATE__[docId] || 100);

    const handler = (e) => {
      const t = e && e.target ? e.target : null;
      if (!t) return;
      if (t.tagName !== "INPUT") return;
      const type = (t.getAttribute("type") || "").toLowerCase();
      if (type !== "range" && type !== "number") return;
      const p = clampZoomPercent(t.value);
      setControlValues(container, p);
      applyZoom(docId, p);
    };

    container.addEventListener("input", handler, true);
    container.addEventListener("change", handler, true);
    return true;
  }

  function wireAllZoomControls() {
    // Primary path: bind by elem_id wrappers.
    const aOk = wireZoomControl("a", "zoom-a-slider");
    const bOk = wireZoomControl("b", "zoom-b-slider");
    return aOk || bOk;
  }

  // Fallback: event delegation if elem_id wrappers are not present in DOM snapshot.
  function installZoomDelegationOnce() {
    if (window.__GLOBAL_ZOOM_DELEGATION__) return;
    window.__GLOBAL_ZOOM_DELEGATION__ = true;

    document.addEventListener(
      "input",
      (e) => {
        const t = e && e.target ? e.target : null;
        if (!t || t.tagName !== "INPUT") return;
        const type = (t.getAttribute("type") || "").toLowerCase();
        if (type !== "range" && type !== "number") return;

        const wrap = t.closest("#zoom-a-slider, #zoom-b-slider");
        if (wrap) {
          const docId = wrap.id === "zoom-a-slider" ? "a" : "b";
          const p = clampZoomPercent(t.value);
          setControlValues(wrap, p);
          applyZoom(docId, p);
          return;
        }
      },
      true
    );
  }

  // Initialize zoom wiring + rewire after Gradio re-renders.
  installZoomDelegationOnce();
  wireAllZoomControls();

  // Reapply zoom on DOM updates (e.g. compare, toggle viewer).
  const zoomObserver = new MutationObserver(() => {
    wireAllZoomControls();
    applyZoom("a", window.__GLOBAL_ZOOM_STATE__.a || 100);
    applyZoom("b", window.__GLOBAL_ZOOM_STATE__.b || 100);
  });
  zoomObserver.observe(document.body, { childList: true, subtree: true });
}
"""


def build_upload_row():
    """Build file upload components."""
    with gr.Row():
        doc1 = gr.File(label="Document A (PDF)", file_types=[".pdf"])
        doc2 = gr.File(label="Document B (PDF)", file_types=[".pdf"])
    return doc1, doc2


def build_status_area():
    """Build status display area."""
    with gr.Row():
        status = gr.Markdown("### Ready to compare documents")
    return status


def build_parameters_panel():
    """Build collapsible parameters panel with sensitivity threshold and options."""
    with gr.Accordion("⚙️ Comparison Parameters", open=False) as params_accordion:
        sensitivity_threshold = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=settings.text_similarity_threshold if settings.text_similarity_threshold is not None else 0.82,
            step=0.01,
            label="Sensitivity Threshold",
            info="Higher = more sensitive (detects minor changes), Lower = less sensitive (only major changes)",
        )
        
        scanned_document_mode = gr.Checkbox(
            label="Scanned Document Mode",
            value=False,
            info="Indicates both PDFs are scanned images. Prioritizes OCR and visual diff methods.",
        )
        
        force_ocr = gr.Checkbox(
            label="OCR Enhancement (Hybrid, safe for digital PDFs)",
            value=False,
            info="Runs native extraction + OCR and merges with a safety gate (prevents DeepSeek hallucinations from overwriting native text).",
        )
        
        # UX: Keep the label short and non-redundant; put details in helper text.
        show_heatmap = gr.Checkbox(
            label="Heatmap overlay",
            value=False,
            info="Highlights detected changes on the rendered page images. Turn off for faster renders.",
            elem_id="heatmap-overlay-toggle",
        )
        
        # OCR Engine Selection (used for scanned mode and hybrid OCR)
        # UX: DeepSeek OCR is intentionally disabled in this UI (unstable availability across machines).
        ocr_engine_options = ["paddle", "tesseract"]
        ocr_engine = gr.Dropdown(
            choices=ocr_engine_options,
            # Default to configured engine (macOS typically wants PaddleOCR).
            value=getattr(settings, "ocr_engine", "paddle") or "paddle",
            label="OCR Engine",
            info="Select OCR engine. PaddleOCR is fast/accurate. Tesseract is legacy.",
            visible=True,
        )

        gr.Markdown("### Model Overrides")
        text_model = gr.Textbox(
            label="Text Embedding Model",
            value=settings.sentence_transformer_model,
            info="SentenceTransformer model name or local path.",
        )
        layout_model = gr.Textbox(
            label="Layout Model Path",
            value=settings.yolo_layout_model_name,
            info="YOLO/DocLayout-YOLO model path or name.",
        )
    
    return (
        params_accordion,
        sensitivity_threshold,
        scanned_document_mode,
        force_ocr,
        show_heatmap,
        ocr_engine,
        text_model,
        layout_model,
    )


def build_performance_panel():
    """Build collapsible performance indicators panel."""
    with gr.Accordion("📊 Performance & System Info", open=False) as perf_accordion:
        performance_info = gr.Markdown("### Performance metrics will appear here after comparison")
        
        # System info
        try:
            gpu_available = is_cuda_available()
            system_info = f"**System:** {'GPU (CUDA)' if gpu_available else 'CPU'}"
        except Exception:
            system_info = "**System:** Unknown"
        
        system_info_display = gr.Markdown(system_info)
    
    return perf_accordion, performance_info, system_info_display


def build_comparison_interface() -> gr.Blocks:
    """Build the complete Gradio interface for document comparison."""
    with gr.Blocks(title="AI Document Comparator") as demo:
        # Premium Design System CSS with Grade A styling
        gr.HTML("""
        <style>
        /* =====================================================
           PREMIUM DESIGN SYSTEM - Grade A Quality
           Inspired by modern web design best practices
           WCAG AA+ Compliant | Glassmorphism | Smooth Animations
           ===================================================== */
        
        /* === CSS Custom Properties (Design Tokens) === */
        :root {
            /* Color Palette - HSL for easy manipulation */
            --color-primary: hsl(190, 95%, 50%);
            --color-primary-light: hsl(190, 95%, 60%);
            --color-primary-dark: hsl(190, 95%, 40%);
            
            --color-accent: hsl(280, 70%, 60%);
            --color-accent-light: hsl(280, 70%, 70%);
            
            --color-success: hsl(150, 70%, 50%);
            --color-warning: hsl(45, 100%, 60%);
            --color-danger: hsl(0, 80%, 60%);
            
            /* Dark Theme Background */
            --color-bg-primary: hsl(220, 15%, 10%);
            --color-bg-secondary: hsl(220, 15%, 15%);
            --color-bg-tertiary: hsl(220, 15%, 20%);
            
            /* Text Colors (WCAG AA+ compliant) */
            --color-text-primary: hsl(0, 0%, 95%);
            --color-text-secondary: hsl(220, 10%, 70%);
            --color-text-muted: hsl(220, 10%, 50%);
            
            /* Border Colors */
            --color-border-light: rgba(148, 163, 184, 0.2);
            --color-border-medium: rgba(148, 163, 184, 0.4);
            
            /* Typography Scale */
            --font-family-base: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            --font-family-mono: 'JetBrains Mono', 'Fira Code', monospace;
            
            --font-size-xs: 0.75rem;    /* 12px */
            --font-size-sm: 0.875rem;   /* 14px */
            --font-size-base: 1rem;     /* 16px */
            --font-size-lg: 1.125rem;   /* 18px */
            --font-size-xl: 1.25rem;    /* 20px */
            --font-size-2xl: 1.5rem;    /* 24px */
            --font-size-3xl: 2rem;      /* 32px */
            
            /* Spacing Scale (8px grid system) */
            --spacing-1: 0.5rem;   /* 8px */
            --spacing-2: 1rem;     /* 16px */
            --spacing-3: 1.5rem;   /* 24px */
            --spacing-4: 2rem;     /* 32px */
            --spacing-5: 2.5rem;   /* 40px */
            --spacing-6: 3rem;     /* 48px */
            
            /* Border Radius */
            --radius-sm: 0.375rem;  /* 6px */
            --radius-md: 0.5rem;    /* 8px */
            --radius-lg: 0.75rem;   /* 12px */
            --radius-xl: 1rem;      /* 16px */
            
            /* Shadows */
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
            --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.4);
            
            /* Transitions */
            --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-base: 250ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-slow: 350ms cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        /* === Global Improvements === */
        * {
            box-sizing: border-box;
        }
        
        body, .gradio-container {
            font-family: var(--font-family-base) !important;
            background: var(--color-bg-primary) !important;
            color: var(--color-text-primary) !important;
        }
        
        /* === Enhanced Gallery Styles === */
        .pdf-gallery {
            scroll-behavior: smooth;
            background: var(--color-bg-secondary);
            border-radius: var(--radius-lg);
            padding: var(--spacing-2);
            border: 1px solid var(--color-border-light);
        }

        /* Allow panning when zoomed > 100% */
        .pdf-gallery .gallery {
            overflow: auto !important;
        }

        .pdf-gallery img,
        .pdf-gallery canvas {
            max-width: none !important;
        }
        
        /* =====================================================
           PDF PAGE FIT FIX (Document A + B)
           Gradio Gallery renders images as square thumbnails by default, which
           crops PDF pages. Force "full page" rendering with responsive sizing.
           ===================================================== */
        #gallery-a .grid-wrap,
        #gallery-b .grid-wrap {
            /* Override inline height coming from `gr.Gallery(height=...)` */
            height: 80vh !important;
            max-height: 1000px !important;
            min-height: 600px !important;
            width: 100% !important;
        }

        #gallery-a .gallery-item,
        #gallery-b .gallery-item {
            width: 100% !important;
            height: auto !important;
        }

        #gallery-a button.thumbnail-item,
        #gallery-b button.thumbnail-item {
            width: 100% !important;
            /* Give the image a stable box so object-fit:contain can fully show the page */
            height: calc(80vh - 140px) !important;
            max-height: 900px !important;
            min-height: 520px !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            overflow: hidden !important;
        }

        #gallery-a button.thumbnail-item > img,
        #gallery-b button.thumbnail-item > img {
            width: 100% !important;
            height: 100% !important;
            object-fit: contain !important;
            display: block !important;
        }

        .pdf-gallery img {
            border: 2px solid transparent;
            transition: border-color var(--transition-base), 
                        transform var(--transition-base),
                        box-shadow var(--transition-base);
            border-radius: var(--radius-md);
        }
        
        .pdf-gallery img:hover {
            transform: translateY(-4px) scale(1.01);
            box-shadow: var(--shadow-xl);
        }
        
        .pdf-gallery img.highlighted {
            border-color: var(--color-primary) !important;
            box-shadow: 0 0 20px rgba(6, 182, 212, 0.6);
            animation: pulseGlow 2s ease-in-out infinite;
        }
        
        @keyframes pulseGlow {
            0%, 100% { box-shadow: 0 0 20px rgba(6, 182, 212, 0.6); }
            50% { box-shadow: 0 0 30px rgba(6, 182, 212, 0.9); }
        }
        
        /* === Synchronized Viewer Styles === */
        .sync-viewer-pdf {
            width: 100%;
            height: 800px;
            border-radius: var(--radius-lg);
            overflow: hidden;
            box-shadow: var(--shadow-xl);
            border: 1px solid var(--color-border-light);
        }
        
        #sync-viewer-container {
            display: flex;
            gap: var(--spacing-4);
            padding: var(--spacing-3);
            background: var(--color-bg-secondary);
            border-radius: var(--radius-xl);
        }
        
        #sync-viewer-container > div {
            flex: 1;
            min-width: 0;
        }
        
        /* === Button Enhancements === */
        button, .gradio-button {
            font-family: var(--font-family-base) !important;
            font-weight: 600;
            border-radius: var(--radius-md) !important;
            transition: all var(--transition-base) !important;
            position: relative;
            overflow: hidden;
        }
        
        button::before, .gradio-button::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.2);
            transform: translate(-50%, -50%);
            transition: width var(--transition-base), height var(--transition-base);
        }
        
        button:hover::before, .gradio-button:hover::before {
            width: 300px;
            height: 300px;
        }
        
        button:hover, .gradio-button:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg) !important;
        }
        
        button:active, .gradio-button:active {
            transform: translateY(0);
        }
        
        /* Primary Button */
        button[variant="primary"], .primary {
            background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-primary-dark) 100%) !important;
            border: none !important;
            color: white !important;
        }
        
        /* === Checkbox Enhancements === */
        input[type="checkbox"] {
            width: 20px !important;
            height: 20px !important;
            cursor: pointer;
            transition: all var(--transition-fast);
        }
        
        input[type="checkbox"]:checked {
            accent-color: var(--color-primary);
        }
        
        input[type="checkbox"]:focus {
            outline: 2px solid var(--color-primary);
            outline-offset: 2px;
        }
        
        /* === Markdown Enhancements === */
        .markdown h1 {
            font-size: var(--font-size-3xl);
            font-weight: 700;
            background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-accent) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: var(--spacing-2);
        }
        
        .markdown h2, .markdown h3 {
            color: var(--color-text-primary);
            font-weight: 600;
            margin-top: var(--spacing-3);
            margin-bottom: var(--spacing-2);
        }
        
        .markdown p {
            color: var(--color-text-secondary);
            line-height: 1.7;
        }
        
        /* === Dataframe/Table Enhancements === */
        .dataframe, table {
            border-radius: var(--radius-lg) !important;
            overflow: hidden !important;
            background: var(--color-bg-secondary) !important;
            border: 1px solid var(--color-border-light) !important;
        }
        
        .dataframe thead, table thead {
            background: var(--color-bg-tertiary) !important;
            font-weight: 600;
        }
        
        .dataframe th, table th {
            color: var(--color-primary) !important;
            padding: var(--spacing-2) !important;
            text-transform: uppercase;
            font-size: var(--font-size-xs);
            letter-spacing: 0.05em;
        }
        
        .dataframe td, table td {
            padding: var(--spacing-2) !important;
            color: var(--color-text-secondary) !important;
            border-bottom: 1px solid var(--color-border-light) !important;
        }
        
        .dataframe tr:hover, table tr:hover {
            background: rgba(6, 182, 212, 0.1) !important;
            cursor: pointer;
        }
        
        /* === Dropdown Enhancements === */
        select, .dropdown {
            background: var(--color-bg-secondary) !important;
            border: 1px solid var(--color-border-medium) !important;
            border-radius: var(--radius-md) !important;
            color: var(--color-text-primary) !important;
            padding: var(--spacing-2) !important;
            font-family: var(--font-family-base) !important;
            transition: all var(--transition-base);
        }
        
        select:hover, .dropdown:hover {
            border-color: var(--color-primary);
            box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.1);
        }
        
        select:focus, .dropdown:focus {
            outline: none;
            border-color: var(--color-primary);
            box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.2);
        }
        
        /* === File Upload Enhancements === */
        .file-upload {
            border: 2px dashed var(--color-border-medium) !important;
            border-radius: var(--radius-lg) !important;
            background: var(--color-bg-secondary) !important;
            transition: all var(--transition-base);
            padding: var(--spacing-4) !important;
        }
        
        .file-upload:hover {
            border-color: var(--color-primary);
            background: rgba(6, 182, 212, 0.05);
            transform: translateY(-2px);
        }
        
        /* === Loading States === */
        @keyframes shimmer {
            0% { background-position: -1000px 0; }
            100% { background-position: 1000px 0; }
        }
        
        .loading-skeleton {
            background: linear-gradient(90deg, 
                var(--color-bg-secondary) 25%, 
                var(--color-bg-tertiary) 50%, 
                var(--color-bg-secondary) 75%);
            background-size: 2000px 100%;
            animation: shimmer 2s infinite;
            border-radius: var(--radius-md);
        }
        
        /* === Accessibility Enhancements === */
        *:focus-visible {
            outline: 2px solid var(--color-primary) !important;
            outline-offset: 2px !important;
        }
        
        /* Screen reader only content */
        .sr-only {
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border-width: 0;
        }
        
        /* === Responsive Design === */
        @media (max-width: 768px) {
            #sync-viewer-container {
                flex-direction: column;
            }
            
            :root {
                --font-size-3xl: 1.75rem;
                --font-size-2xl: 1.375rem;
            }
        }
        
        /* === Scrollbar Styling === */
        ::-webkit-scrollbar {
            width: 12px;
            height: 12px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--color-bg-secondary);
            border-radius: var(--radius-md);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--color-bg-tertiary);
            border-radius: var(--radius-md);
            border: 2px solid var(--color-bg-secondary);
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--color-primary);
        }
        
        /* === Animation Utilities === */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes slideInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes scaleIn {
            from {
                opacity: 0;
                transform: scale(0.9);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }
        
        .animate-fadeIn { animation: fadeIn var(--transition-base) ease-out; }
        .animate-slideInUp { animation: slideInUp var(--transition-base) ease-out; }
        .animate-scaleIn { animation: scaleIn var(--transition-base) ease-out; }

        /* =====================================================
           Heatmap toggle (UX polish)
           Keep helper text compact and readable.
           ===================================================== */
        #heatmap-overlay-toggle .wrap,
        #heatmap-overlay-toggle .block {
            padding: 10px 12px !important;
        }
        #heatmap-overlay-toggle .info,
        #heatmap-overlay-toggle [class*="info"] {
            font-size: 0.85rem !important;
            line-height: 1.25rem !important;
            color: var(--color-text-secondary) !important;
            margin-top: 4px !important;
        }
        
        </style>
        """, visible=False)
        
        # Enhanced scroll synchronization JavaScript
        gr.HTML("""
        <script>
        // =====================================================
        // PREMIUM SCROLL SYNC & NAVIGATION
        // Debounced sync | Visual indicators | Keyboard shortcuts
        // =====================================================
        
        (function() {
            let activeGallery = null;
            let syncIndicator = null;
            
            // Debounce utility for smooth scroll sync
            function debounce(func, wait) {
                let timeout;
                return function executedFunction(...args) {
                    const later = () => {
                        clearTimeout(timeout);
                        func(...args);
                    };
                    clearTimeout(timeout);
                    timeout = setTimeout(later, wait);
                };
            }
            
            // Scroll to page with smooth animation and highlight
            function scrollToPage(galleryId, pageIndex) {
                const gallery = document.querySelector(`#${galleryId} .gallery`);
                if (!gallery) return;
                
                const images = gallery.querySelectorAll('img');
                if (images[pageIndex]) {
                    // Show sync indicator
                    showSyncIndicator('Navigating to page ' + (pageIndex + 1));
                    
                    // Smooth scroll
                    images[pageIndex].scrollIntoView({ 
                        behavior: 'smooth', 
                        block: 'center' 
                    });
                    
                    // Add highlight animation
                    images.forEach(img => img.classList.remove('highlighted'));
                    images[pageIndex].classList.add('highlighted');
                    
                    // Remove highlight after animation
                    setTimeout(() => {
                        images[pageIndex].classList.remove('highlighted');
                        hideSyncIndicator();
                    }, 2000);
                    
                    activeGallery = galleryId;
                }
            }
            
            // Visual sync indicator
            function createSyncIndicator() {
                const indicator = document.createElement('div');
                indicator.id = 'sync-indicator';
                indicator.setAttribute('role', 'status');
                indicator.setAttribute('aria-live', 'polite');
                indicator.style.cssText = `
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: linear-gradient(135deg, hsl(190, 95%, 50%), hsl(280, 70%, 60%));
                    color: white;
                    padding: 12px 24px;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: 600;
                    font-family: 'Inter', sans-serif;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                    z-index: 9999;
                    opacity: 0;
                    transform: translateY(-20px);
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                    pointer-events: none;
                `;
                document.body.appendChild(indicator);
                syncIndicator = indicator;
            }
            
            function showSyncIndicator(message) {
                if (!syncIndicator) createSyncIndicator();
                syncIndicator.textContent = message;
                syncIndicator.style.opacity = '1';
                syncIndicator.style.transform = 'translateY(0)';
            }
            
            function hideSyncIndicator() {
                if (syncIndicator) {
                    syncIndicator.style.opacity = '0';
                    syncIndicator.style.transform = 'translateY(-20px)';
                }
            }
            
            // Keyboard shortcuts for navigation
            document.addEventListener('keydown', function(e) {
                // Only activate if not in input/textarea
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
                
                const galleries = ['gallery-a', 'gallery-b'];
                const currentGallery = activeGallery || galleries[0];
                const gallery = document.querySelector(`#${currentGallery} .gallery`);
                
                if (!gallery) return;
                
                const images = Array.from(gallery.querySelectorAll('img'));
                const currentIndex = images.findIndex(img => img.classList.contains('highlighted'));
                
                switch(e.key.toLowerCase()) {
                    case 'j': // Next page
                        e.preventDefault();
                        if (currentIndex < images.length - 1) {
                            scrollToPage(currentGallery, currentIndex + 1);
                            showSyncIndicator('⬇️ Next page');
                        }
                        break;
                    case 'k': // Previous page
                        e.preventDefault();
                        if (currentIndex > 0) {
                            scrollToPage(currentGallery, currentIndex - 1);
                            showSyncIndicator('⬆️ Previous page');
                        }
                        break;
                    case 'escape':
                        // Clear all highlights
                        images.forEach(img => img.classList.remove('highlighted'));
                        break;
                }
            });
            
            // Make function globally accessible
            window.scrollToPage = scrollToPage;
            
            // Scroll synchronization between galleries
            let scrollSyncEnabled = true;
            let isScrolling = false;
            
            function syncGalleryScroll(sourceGallery, targetGallery) {
                if (!scrollSyncEnabled || isScrolling) return;
                
                isScrolling = true;
                const source = document.querySelector(`#${sourceGallery} .gallery`);
                const target = document.querySelector(`#${targetGallery} .gallery`);
                
                if (source && target) {
                    const scrollRatio = source.scrollTop / (source.scrollHeight - source.clientHeight);
                    const targetScroll = scrollRatio * (target.scrollHeight - target.clientHeight);
                    target.scrollTop = targetScroll;
                }
                
                setTimeout(() => { isScrolling = false; }, 50);
            }
            
            // Listen for scroll sync toggle
            document.addEventListener('change', function(e) {
                if (e.target.type === 'checkbox' && 
                    e.target.getAttribute('aria-label')?.includes('Synchronize Scrolling')) {
                    scrollSyncEnabled = e.target.checked;
                }
            });
            
            // Set up scroll listeners for galleries
            function setupScrollSync() {
                const galleryA = document.querySelector('#gallery-a .gallery');
                const galleryB = document.querySelector('#gallery-b .gallery');
                
                if (galleryA && galleryB) {
                    galleryA.addEventListener('scroll', () => {
                        if (scrollSyncEnabled) syncGalleryScroll('gallery-a', 'gallery-b');
                    });
                    
                    galleryB.addEventListener('scroll', () => {
                        if (scrollSyncEnabled) syncGalleryScroll('gallery-b', 'gallery-a');
                    });
                }
            }
            
            // Initialize on DOM ready
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', function() {
                    createSyncIndicator();
                    setupScrollSync();
                });
            } else {
                createSyncIndicator();
                setupScrollSync();
            }
            
            // Re-setup after Gradio updates
            const observer = new MutationObserver(function() {
                setupScrollSync();
            });
            observer.observe(document.body, { childList: true, subtree: true });
            
            console.log('✨ Premium scroll sync initialized with keyboard shortcuts (j/k)');
        })();
        
        // =====================================================
        // CLICKABLE DATAFRAME ROWS
        // Make diff list rows clickable to jump to changes
        // =====================================================
        (function() {
            function makeDataframeClickable() {
                const dataframe = document.querySelector('#diff-list-dataframe table');
                if (!dataframe) return;
                
                const rows = dataframe.querySelectorAll('tbody tr');
                rows.forEach((row, index) => {
                    row.style.cursor = 'pointer';
                    row.addEventListener('click', function() {
                        // Remove previous selection
                        rows.forEach(r => r.style.backgroundColor = '');
                        // Highlight selected row
                        this.style.backgroundColor = 'rgba(6, 182, 212, 0.2)';
                        
                        // Trigger dropdown selection by finding the corresponding option
                        // The dropdown value should match the diff index
                        // We'll use a custom event or directly update the dropdown
                        const dropdown = document.querySelector('select[aria-label*="Jump to Change"]');
                        if (dropdown) {
                            // Find option that corresponds to this row index
                            const options = Array.from(dropdown.options);
                            // Try to match by page number from first cell
                            const pageCell = this.querySelector('td:first-child');
                            if (pageCell) {
                                const pageNum = pageCell.textContent.trim();
                                // Find option containing this page number
                                const matchingOption = options.find(opt => 
                                    opt.textContent.includes(`Page ${pageNum}:`)
                                );
                                if (matchingOption) {
                                    dropdown.value = matchingOption.value;
                                    dropdown.dispatchEvent(new Event('change', { bubbles: true }));
                                }
                            }
                        }
                    });
                    
                    // Add hover effect
                    row.addEventListener('mouseenter', function() {
                        if (this.style.backgroundColor !== 'rgba(6, 182, 212, 0.2)') {
                            this.style.backgroundColor = 'rgba(6, 182, 212, 0.1)';
                        }
                    });
                    row.addEventListener('mouseleave', function() {
                        if (this.style.backgroundColor !== 'rgba(6, 182, 212, 0.2)') {
                            this.style.backgroundColor = '';
                        }
                    });
                });
            }
            
            // Initialize on load and after updates
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', makeDataframeClickable);
            } else {
                makeDataframeClickable();
            }
            
            // Re-initialize after Gradio updates
            const observer = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    if (mutation.addedNodes.length) {
                        makeDataframeClickable();
                    }
                });
            });
            
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
            
            console.log('✨ Clickable dataframe rows initialized');
        })();
        
        // =====================================================
        // ZOOM CONTROLS - Direct event delegation for immediate response
        // Apply zoom to gallery images and sync viewer (pure client-side, no refresh)
        // =====================================================
        (function() {
            // Store zoom levels
            const zoomLevels = { 'a': 100, 'b': 100 };
            
            function applyZoomToGallery(galleryId, zoomLevel) {
                const percent = Math.max(25, Math.min(300, Number(zoomLevel) || 100));
                const factor = percent / 100;

                // Try gallery view first
                const galleryRoot = document.getElementById(galleryId);
                if (galleryRoot) {
                    // Find the actual gallery container
                    let gallery = galleryRoot.querySelector('.gallery');
                    if (!gallery) {
                        gallery = galleryRoot.querySelector('div[class*="gallery"]');
                    }
                    if (!gallery) {
                        const scrollable = Array.from(galleryRoot.querySelectorAll('div')).find(
                            div => getComputedStyle(div).overflow !== 'visible'
                        );
                        if (scrollable) gallery = scrollable;
                    }
                    if (!gallery) {
                        gallery = galleryRoot;
                    }

                    // Find all images in the gallery
                    const images = gallery.querySelectorAll('img, canvas');
                    
                    if (images.length > 0) {
                        gallery.style.overflow = 'auto';
                        gallery.style.scrollBehavior = 'smooth';
                        
                        // Try CSS zoom first (works in Chrome/Safari/Edge)
                        gallery.style.zoom = String(factor);
                        
                        // Verify zoom was applied
                        const zoomApplied = gallery.style.zoom === String(factor) || 
                                           getComputedStyle(gallery).zoom === String(factor);
                        
                        if (!zoomApplied) {
                            // CSS zoom not supported, use transform on images
                            gallery.style.zoom = '';
                            images.forEach(img => {
                                if (img.tagName === 'IMG' && !img.complete) {
                                    img.addEventListener('load', function onLoad() {
                                        img.removeEventListener('load', onLoad);
                                        applyZoomToImage(img, factor);
                                    }, { once: true });
                                } else {
                                    applyZoomToImage(img, factor);
                                }
                            });
                        } else {
                            // Clear conflicting styles when using CSS zoom
                            images.forEach(img => {
                                img.style.width = '';
                                img.style.height = '';
                                img.style.maxWidth = '';
                                img.style.transform = '';
                                img.style.transformOrigin = '';
                            });
                        }
                        return;
                    }
                }
                
                // Try sync viewer (different structure)
                const viewerId = galleryId === 'gallery-a' ? 'a' : 'b';
                const syncViewer = document.getElementById(`sync-img-viewer-${viewerId}`);
                if (syncViewer) {
                    const images = syncViewer.querySelectorAll('.sync-page-img, img');
                    if (images.length > 0) {
                        // Apply zoom to sync viewer container
                        syncViewer.style.overflow = 'auto';
                        syncViewer.style.scrollBehavior = 'smooth';
                        
                        // Try CSS zoom first
                        syncViewer.style.zoom = String(factor);
                        const zoomApplied = syncViewer.style.zoom === String(factor) || 
                                           getComputedStyle(syncViewer).zoom === String(factor);
                        
                        if (!zoomApplied) {
                            // Use transform for each image
                            syncViewer.style.zoom = '';
                            images.forEach(img => {
                                applyZoomToImage(img, factor);
                            });
                        } else {
                            // Clear conflicting styles
                            images.forEach(img => {
                                img.style.width = '';
                                img.style.height = '';
                                img.style.maxWidth = '';
                                img.style.transform = '';
                                img.style.transformOrigin = '';
                            });
                        }
                    }
                }
            }
            
            function applyZoomToImage(img, factor) {
                // Remove any previous zoom styles
                img.style.width = '';
                img.style.height = '';
                img.style.maxWidth = 'none';
                img.style.minWidth = 'none';
                
                // Use transform for zoom (works everywhere)
                img.style.transform = `scale(${factor})`;
                img.style.transformOrigin = 'top left';
                
                // Ensure parent container can handle the scaled content
                let parent = img.parentElement;
                while (parent && parent !== document.body) {
                    const overflow = getComputedStyle(parent).overflow;
                    if (overflow === 'hidden' || overflow === 'visible') {
                        parent.style.overflow = 'auto';
                    }
                    parent = parent.parentElement;
                }
            }

            function _tryParseNumber(v) {
                const n = parseFloat(v);
                return Number.isFinite(n) ? n : null;
            }

            function resolveZoomContext(target) {
                if (!target || !target.closest) return null;

                // Prefer explicit containers (we set elem_id on the Slider wrapper)
                const container = target.closest('#zoom-a-slider, #zoom-b-slider');
                if (container) {
                    const isA = container.id.includes('zoom-a');
                    const docId = isA ? 'a' : 'b';
                    const galleryId = isA ? 'gallery-a' : 'gallery-b';

                    // Extract value from whichever element fired the event
                    let zoomLevel = _tryParseNumber(target.value);
                    if (zoomLevel === null) zoomLevel = _tryParseNumber(target.getAttribute?.('aria-valuenow'));

                    // Fallbacks: read from an input inside the container (Gradio renders a number input + slider)
                    if (zoomLevel === null) {
                        const numberInput = container.querySelector('input[type="number"], input');
                        if (numberInput) zoomLevel = _tryParseNumber(numberInput.value);
                    }
                    if (zoomLevel === null) {
                        const rangeInput = container.querySelector('input[type="range"]');
                        if (rangeInput) zoomLevel = _tryParseNumber(rangeInput.value);
                    }

                    return { docId, galleryId, zoomLevel: zoomLevel ?? (zoomLevels[docId] ?? 100) };
                }

                // Back-compat fallback: try to infer by IDs/text context
                const sliderId = target.id || target.closest('[id*="zoom"]')?.id || '';
                let galleryId = null;
                let docId = null;

                if (sliderId.includes('zoom-a') || sliderId.includes('zoom-a-slider')) {
                    galleryId = 'gallery-a';
                    docId = 'a';
                } else if (sliderId.includes('zoom-b') || sliderId.includes('zoom-b-slider')) {
                    galleryId = 'gallery-b';
                    docId = 'b';
                } else {
                    let parent = target.parentElement;
                    for (let i = 0; i < 15 && parent; i++) {
                        const text = (parent.textContent || '').toLowerCase();
                        if (text.includes('document a') || text.includes('doc a')) {
                            galleryId = 'gallery-a';
                            docId = 'a';
                            break;
                        } else if (text.includes('document b') || text.includes('doc b')) {
                            galleryId = 'gallery-b';
                            docId = 'b';
                            break;
                        }
                        parent = parent.parentElement;
                    }
                }

                if (galleryId && docId) {
                    const zoomLevel = _tryParseNumber(target.value) ?? 100;
                    return { docId, galleryId, zoomLevel };
                }

                return null;
            }

            function handleZoomEvent(e) {
                const ctx = resolveZoomContext(e.target);
                if (!ctx) return;
                zoomLevels[ctx.docId] = ctx.zoomLevel;
                applyZoomToGallery(ctx.galleryId, ctx.zoomLevel);
            }

            // Use capture phase for immediate response. We listen to several events because
            // Gradio sliders may not be plain <input type="range"> elements in all versions.
            ['input', 'change', 'keyup', 'pointerup'].forEach((ev) => {
                document.addEventListener(ev, handleZoomEvent, true);
            });

            // Function to reapply zoom when content changes
            function reapplyZoom(docId) {
                const galleryId = docId === 'a' ? 'gallery-a' : 'gallery-b';
                const zoomLevel = zoomLevels[docId] || 100;
                applyZoomToGallery(galleryId, zoomLevel);
            }

            // Watch for gallery/viewer updates and reapply zoom
            function setupWatchers() {
                ['a', 'b'].forEach(docId => {
                    const galleryId = docId === 'a' ? 'gallery-a' : 'gallery-b';
                    const viewerId = docId;
                    const gallery = document.getElementById(galleryId);
                    const syncViewer = document.getElementById(`sync-img-viewer-${viewerId}`);
                    
                    [gallery, syncViewer].filter(Boolean).forEach(target => {
                        const observer = new MutationObserver(() => {
                            // Reapply zoom when content changes
                            setTimeout(() => reapplyZoom(docId), 50);
                        });
                        observer.observe(target, { 
                            childList: true, 
                            subtree: true,
                            attributes: false
                        });
                    });
                });
            }

            // Initialize watchers
            function init() {
                setupWatchers();
                // Reapply any existing zoom levels
                ['a', 'b'].forEach(docId => {
                    reapplyZoom(docId);
                });
            }

            // Initialize when DOM is ready
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', init);
            } else {
                init();
            }

            // Re-setup watchers on DOM changes
            const domObserver = new MutationObserver(() => {
                setupWatchers();
            });
            domObserver.observe(document.body, { childList: true, subtree: true });

            // Listen for Gradio events
            document.addEventListener('gradio:loaded', () => {
                setTimeout(init, 100);
            });

            console.log('✨ Zoom controls loaded with event delegation (immediate response)');
        })();
        </script>
        """, visible=False)

        # Sync viewer JS is loaded via demo.load(..., js=...) for reliability.

        gr.Markdown("# 🔍 AI Document Comparison System")
        gr.Markdown("""
        Upload two PDF documents to compare and visualize differences with AI-powered analysis.
        
        **Keyboard Shortcuts:** Press `j` for next diff | `k` for previous diff | `Esc` to clear highlights
        """)
        
        with gr.Row():
            doc1, doc2 = build_upload_row()
        
        # Parameters Panel (collapsible)
        (
            params_accordion,
            sensitivity_threshold,
            scanned_document_mode,
            force_ocr,
            show_heatmap,
            ocr_engine,
            text_model,
            layout_model,
        ) = build_parameters_panel()
        
        with gr.Row():
            compare_btn = gr.Button("Compare Documents", variant="primary")
            use_sync_viewer = gr.Checkbox(
                label="Use Synchronized Viewer",
                value=False,
                info="Enable advanced synchronized PDF viewer with semantic scrolling",
                elem_id="use-sync-viewer-checkbox",
                interactive=True,
                container=True
            )
        
        status = build_status_area()
        
        # Performance Panel (collapsible)
        perf_accordion, performance_info, system_info_display = build_performance_panel()
        
        # Enhanced split-pane layout with better proportions
        # Standard Gallery view
        with gr.Row(visible=True) as gallery_row:
            with gr.Column(scale=1, min_width=400):
                with gr.Row():
                    gr.Markdown("### Document A")
                    _zoom_a = gr.Slider(
                        minimum=50,
                        maximum=200,
                        value=100,
                        step=10,
                        label="Zoom (%)",
                        elem_id="zoom-a-slider",
                        interactive=True,
                        scale=1,
                        container=False,
                    )
                gallery_a = gr.Gallery(
                    label="Pages",
                    columns=1,
                    height=600,
                    show_label=False,
                    type="numpy",
                    elem_classes=["pdf-gallery"],
                    elem_id="gallery-a",
                )
            with gr.Column(scale=1, min_width=400):
                with gr.Row():
                    gr.Markdown("### Document B")
                    _zoom_b = gr.Slider(
                        minimum=50,
                        maximum=200,
                        value=100,
                        step=10,
                        label="Zoom (%)",
                        elem_id="zoom-b-slider",
                        interactive=True,
                        scale=1,
                        container=False,
                    )
                gallery_b = gr.Gallery(
                    label="Pages",
                    columns=1,
                    height=600,
                    show_label=False,
                    type="numpy",
                    elem_classes=["pdf-gallery"],
                    elem_id="gallery-b",
                )
        
        # Synchronized PDF viewer (hidden by default)
        with gr.Row(elem_id="sync-viewer-container", visible=False) as sync_viewer_container:
            with gr.Column(scale=1):
                gr.Markdown("### Document A")
                sync_viewer_a = gr.HTML(elem_id="sync-pdf-viewer-a", visible=True)
            with gr.Column(scale=1):
                gr.Markdown("### Document B")
                sync_viewer_b = gr.HTML(elem_id="sync-pdf-viewer-b", visible=True)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Diff Navigator")
                
                # Navigation controls
                with gr.Row():
                    prev_diff_btn = gr.Button("◀ Previous Change", scale=1, variant="secondary")
                    next_diff_btn = gr.Button("Next Change ▶", scale=1, variant="secondary")
                    _sync_scrolling = gr.Checkbox(
                        label="Synchronize Scrolling",
                        value=True,
                        scale=1,
                        info="Link/unlink scrolling between Document A and B",
                    )
                
                # Dropdown for selecting a specific diff to jump to
                diff_selector = gr.Dropdown(
                    label="Jump to Change",
                    choices=[],
                    interactive=True,
                    value=None,
                )
                diff_list = gr.Dataframe(
                    label="Detected Changes",
                    headers=["Page", "Type", "Change Type", "Severity", "Description"],
                    interactive=False,
                    wrap=True,
                    elem_id="diff-list-dataframe",
                )
                with gr.Row():
                    filter_content = gr.Checkbox(label="Show Content Changes", value=True)
                    filter_formatting = gr.Checkbox(label="Show Formatting Changes", value=True)
                    filter_layout = gr.Checkbox(label="Show Layout Changes", value=True)
                    filter_visual = gr.Checkbox(label="Show Visual Changes", value=True)
        
        with gr.Row():
            export_json_btn = gr.Button("Export JSON", scale=1)
            export_pdf_btn = gr.Button("Export PDF Report", scale=1)
        
        # Download file outputs for exports
        with gr.Row():
            json_download = gr.File(
                label="JSON Export",
                visible=False,
                interactive=False,
            )
            pdf_download = gr.File(
                label="PDF Report",
                visible=False,
                interactive=False,
            )
        
        # Store comparison result and selected diff
        comparison_result = gr.State()
        selected_diff_index = gr.State(value=None)
        
        # Store gallery data for scrolling
        gallery_a_data_state = gr.State()
        gallery_b_data_state = gr.State()
        pdf_a_path_state = gr.State()
        pdf_b_path_state = gr.State()
        alignment_data_state = gr.State()
        
        # Zoom is handled client-side (see injected JS listeners).
        # Avoid backend slider events because Gradio's slider preprocess can crash
        # if the frontend ever submits `null` for a slider value.
        
        # Store performance metrics
        performance_metrics_state = gr.State(value={})
        
        def get_severity(confidence: float) -> str:
            """Get severity indicator based on confidence score."""
            if confidence >= 0.7:
                return "🔴 High"
            elif confidence >= 0.4:
                return "🟡 Medium"
            else:
                return "🟢 Low"
        
        def toggle_viewer_visibility(use_sync: bool):
            """Toggle between Gallery and Synchronized viewer (visibility only).

            Important: Updating gr.HTML while a container is hidden can be dropped in
            some Gradio versions. We render the sync viewer content in a chained `.then()`
            after the container becomes visible.
            """
            use_sync = bool(use_sync) if use_sync is not None else False
            return gr.update(visible=not use_sync), gr.update(visible=use_sync)

        def render_sync_viewer_content(use_sync, result: ComparisonResult, pdf_a_path, pdf_b_path, alignment_data):
            """Render sync viewer HTML (called after sync container is visible)."""
            use_sync = bool(use_sync) if use_sync is not None else False
            if not use_sync:
                # Keep existing HTML (avoid expensive rerender when toggling off)
                return gr.update(), gr.update()

            if not (pdf_a_path and pdf_b_path):
                return (
                    gr.update(value="<div style='color:#94a3b8;padding:20px;text-align:center;'>No PDF loaded. Please compare documents first.</div>"),
                    gr.update(value="<div style='color:#94a3b8;padding:20px;text-align:center;'>No PDF loaded. Please compare documents first.</div>"),
                )

            try:
                from visualization.custom_components.sync_pdf_viewer import (
                    _generate_pdf_html,
                )

                alignment_data_dict = alignment_data if alignment_data else {}

                diffs = None
                try:
                    if result and getattr(result, "diffs", None):
                        diffs = result.diffs
                except Exception:
                    diffs = None

                return (
                    gr.update(value=_generate_pdf_html("a", pdf_a_path, None, alignment_data=alignment_data_dict, diffs=diffs)),
                    gr.update(value=_generate_pdf_html("b", pdf_b_path, None, alignment_data=alignment_data_dict, diffs=diffs)),
                )
            except Exception as e:
                logger.exception("Error generating sync viewer: %s", e)
                return (
                    gr.update(value=f"<div style='color:red;padding:20px;'>Error loading viewer: {str(e)}</div>"),
                    gr.update(value=f"<div style='color:red;padding:20px;'>Error loading viewer: {str(e)}</div>"),
                )
        
        def merge_nearby_diffs(diffs: list, x_threshold: float = 0.05, y_threshold: float = 0.02) -> list:
            """Merge very close diffs to reduce bbox clutter.

            Note: in this codebase, `Diff.bbox` is expected to be a dict in normalized
            coordinates: {"x","y","width","height"}. Older code may provide a
            list/tuple-like bbox. This function supports both.
            """

            def _xywh(bbox):
                if not bbox:
                    return None
                if isinstance(bbox, dict):
                    x = float(bbox.get("x", bbox.get("left", 0.0)) or 0.0)
                    y = float(bbox.get("y", bbox.get("top", 0.0)) or 0.0)
                    w = float(bbox.get("width", bbox.get("w", 0.0)) or 0.0)
                    h = float(bbox.get("height", bbox.get("h", 0.0)) or 0.0)
                    return x, y, w, h
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                return None

            def _bbox_for_sort(diff):
                # Prefer A bbox; fall back to B bbox when A is missing.
                box = _xywh(getattr(diff, "bbox", None))
                if box:
                    return box
                return _xywh(getattr(diff, "bbox_b", None))

            def _sort_key(d):
                xywh = _bbox_for_sort(d)
                if not xywh:
                    return (d.page_num, 1e9, 1e9)
                x, y, _, _ = xywh
                return (d.page_num, y, x)

            def _merge_group(group):
                if len(group) == 1:
                    return group[0]

                first = group[0]
                boxes_a = [_xywh(getattr(g, "bbox", None)) for g in group if _xywh(getattr(g, "bbox", None))]
                boxes_b = [_xywh(getattr(g, "bbox_b", None)) for g in group if _xywh(getattr(g, "bbox_b", None))]

                # If neither side has bboxes, nothing to merge.
                if not boxes_a and not boxes_b:
                    return first

                new_bbox = None
                if boxes_a:
                    min_x = min(x for x, _, _, _ in boxes_a)
                    min_y = min(y for _, y, _, _ in boxes_a)
                    max_x = max(x + w for x, _, w, _ in boxes_a)
                    max_y = max(y + h for _, y, _, h in boxes_a)
                    new_bbox = {
                        "x": float(min_x),
                        "y": float(min_y),
                        "width": float(max_x - min_x),
                        "height": float(max_y - min_y),
                    }

                new_bbox_b = None
                if boxes_b:
                    min_xb = min(x for x, _, _, _ in boxes_b)
                    min_yb = min(y for _, y, _, _ in boxes_b)
                    max_xb = max(x + w for x, _, w, _ in boxes_b)
                    max_yb = max(y + h for _, y, _, h in boxes_b)
                    new_bbox_b = {
                        "x": float(min_xb),
                        "y": float(min_yb),
                        "width": float(max_xb - min_xb),
                        "height": float(max_yb - min_yb),
                    }

                old_texts = [g.old_text for g in group if getattr(g, "old_text", None)]
                new_texts = [g.new_text for g in group if getattr(g, "new_text", None)]

                from comparison.models import Diff

                return Diff(
                    page_num=first.page_num,
                    page_num_b=getattr(first, "page_num_b", None),
                    change_type=first.change_type,
                    diff_type=first.diff_type,
                    bbox=new_bbox,
                    bbox_b=new_bbox_b,
                    old_text=" ".join(old_texts) if old_texts else first.old_text,
                    new_text=" ".join(new_texts) if new_texts else first.new_text,
                    confidence=sum(g.confidence for g in group) / len(group),
                    metadata=first.metadata,
                )

            if not diffs:
                return []

            sorted_diffs = sorted(diffs, key=_sort_key)
            merged: list = []
            current_group: list = []

            for diff in sorted_diffs:
                if not current_group:
                    current_group = [diff]
                    continue

                last = current_group[-1]

                # Only merge within same page and same change/diff type.
                if (
                    diff.page_num == last.page_num
                    and diff.change_type == last.change_type
                    and diff.diff_type == last.diff_type
                ):
                    d_box = _bbox_for_sort(diff)
                    l_box = _bbox_for_sort(last)

                    if d_box and l_box:
                        dx, dy, dw, dh = d_box
                        lx, ly, lw, lh = l_box

                        # Gap between rectangles (0 if overlapping).
                        horizontal_gap = max(0.0, dx - (lx + lw), lx - (dx + dw))
                        vertical_gap = max(0.0, dy - (ly + lh), ly - (dy + dh))

                        same_line = abs(dy - ly) < y_threshold

                        if vertical_gap < y_threshold or (same_line and horizontal_gap < x_threshold):
                            current_group.append(diff)
                            continue

                merged.append(_merge_group(current_group))
                current_group = [diff]

            if current_group:
                merged.append(_merge_group(current_group))

            return merged
        
        def collapse_for_logical_changes(diffs: list, x_tolerance: float = 0.1) -> tuple:
            """Collapse vertically stacked diffs into logical changes for metrics.
            
            Keeps individual diffs for UI highlighting, but returns a separate
            list of "logical changes" for summary/metrics.
            
            Diffs in the same X-column (similar x coordinates) with same type
            are collapsed into one logical change with metadata["bboxes"].
            
            Returns:
                (highlight_diffs, logical_changes): 
                - highlight_diffs: original surgical diffs for visualization
                - logical_changes: collapsed diffs for metrics/summary
            """
            if not diffs:
                return [], []
            
            # highlight_diffs stay as-is for visualization
            highlight_diffs = diffs
            
            def _xywh(bbox):
                if not bbox:
                    return None
                if isinstance(bbox, dict):
                    x = float(bbox.get("x", 0.0) or 0.0)
                    y = float(bbox.get("y", 0.0) or 0.0)
                    w = float(bbox.get("width", 0.0) or 0.0)
                    h = float(bbox.get("height", 0.0) or 0.0)
                    return x, y, w, h
                return None

            def _primary_bbox(diff):
                # Prefer A bbox; fall back to B bbox when A is missing.
                return getattr(diff, "bbox", None) or getattr(diff, "bbox_b", None)
            
            # Group diffs by (page, change_type, diff_type, x_column)
            from collections import defaultdict
            column_groups = defaultdict(list)
            
            for diff in diffs:
                box = _xywh(_primary_bbox(diff))
                if box:
                    x, _, w, _ = box
                    x_center = x + w / 2
                    # Round to tolerance to group similar X positions
                    x_bucket = round(x_center / x_tolerance) * x_tolerance
                else:
                    x_bucket = 0.0
                
                key = (diff.page_num, diff.change_type, diff.diff_type, x_bucket)
                column_groups[key].append(diff)
            
            # Create logical changes (collapsed)
            logical_changes = []
            from comparison.models import Diff
            
            for key, group in column_groups.items():
                if len(group) == 1:
                    # Single diff, no collapse needed
                    logical_changes.append(group[0])
                else:
                    # Collapse: merge bboxes into metadata["bboxes"]
                    first = group[0]
                    all_bboxes = [d.bbox for d in group if d.bbox]
                    
                    # Calculate combined bbox (for fallback/main bbox)
                    boxes = [_xywh(b) for b in all_bboxes if _xywh(b)]
                    if boxes:
                        min_x = min(x for x, _, _, _ in boxes)
                        min_y = min(y for _, y, _, _ in boxes)
                        max_x = max(x + w for x, _, w, _ in boxes)
                        max_y = max(y + h for _, y, _, h in boxes)
                        combined_bbox = {
                            "x": min_x, "y": min_y,
                            "width": max_x - min_x, "height": max_y - min_y
                        }
                    else:
                        combined_bbox = first.bbox
                    
                    # Combine text
                    old_texts = [d.old_text for d in group if d.old_text]
                    new_texts = [d.new_text for d in group if d.new_text]
                    
                    # Create metadata with individual bboxes
                    meta = dict(first.metadata) if first.metadata else {}
                    meta["bboxes"] = all_bboxes
                    meta["collapsed_count"] = len(group)
                    
                    logical_diff = Diff(
                        page_num=first.page_num,
                        page_num_b=getattr(first, "page_num_b", None),
                        change_type=first.change_type,
                        diff_type=first.diff_type,
                        bbox=combined_bbox,
                        bbox_b=getattr(first, "bbox_b", None),
                        old_text=" ".join(old_texts) if old_texts else first.old_text,
                        new_text=" ".join(new_texts) if new_texts else first.new_text,
                        confidence=sum(d.confidence for d in group) / len(group),
                        metadata=meta,
                    )
                    logical_changes.append(logical_diff)
            
            return highlight_diffs, logical_changes
        
        def compare_documents(
            file_a, file_b, show_heat, scanned_mode, 
            force_ocr_mode,
            sensitivity,
            use_sync_viewer_flag,
            selected_ocr_engine,
            selected_text_model,
            selected_layout_model,
        ):
            """Main comparison function."""
            if not file_a or not file_b:
                yield (
                    "Please upload both PDF documents.",
                    None,
                    None,
                    None,
                    gr.update(choices=[], value=None),  # diff_selector
                    None,
                    None,
                    None,
                    None,
                    None,  # sync_viewer_a
                    None,  # sync_viewer_b
                    None,  # pdf_a_path_state
                    None,  # pdf_b_path_state
                    None,  # alignment_data_state
                    None,  # performance_metrics_state
                    "",  # performance_info
                )
                return
            
            try:
                # Clear previous timings
                clear_timings()
                
                # Validate inputs
                path_a = validate_pdf_path(file_a.name)
                path_b = validate_pdf_path(file_b.name)
                
                # Configure settings based on parameters
                from config.settings import settings
                
                # Update sensitivity threshold
                settings.text_similarity_threshold = sensitivity
                
                # Update OCR engine
                ocr_engine_priority_override = None
                if selected_ocr_engine:
                    settings.ocr_engine = selected_ocr_engine
                    # Pass as an explicit override into extraction so scanned-policy defaults
                    # (ocr_scanned_default_chain) stay intact when user doesn't choose an engine.
                    ocr_engine_priority_override = [selected_ocr_engine]
                    logger.info("OCR engine set to: %s", selected_ocr_engine)

                # Update model overrides (optional)
                if selected_text_model and selected_text_model.strip():
                    settings.sentence_transformer_model = selected_text_model.strip()
                    logger.info("Text model set to: %s", settings.sentence_transformer_model)
                if selected_layout_model and selected_layout_model.strip():
                    settings.yolo_layout_model_name = selected_layout_model.strip()
                    logger.info("Layout model set to: %s", settings.yolo_layout_model_name)
                
                # Configure OCR mode:
                # - Scanned mode => OCR only (force OCR, typical for image-only PDFs)
                # - Hybrid toggle => hybrid (native + OCR with safety gate; does NOT force OCR-only on digital PDFs)
                # - Otherwise => auto (native extraction for digital PDFs; OCR only for scanned PDFs)
                if scanned_mode:
                    settings.use_ocr_for_all_documents = True
                    settings.ocr_enhancement_mode = "ocr_only"
                    use_ocr = True
                    status_msg = "Extracting scanned documents with OCR (OCR-only)..."
                elif force_ocr_mode:
                    settings.use_ocr_for_all_documents = False
                    settings.ocr_enhancement_mode = "hybrid"
                    use_ocr = False  # do NOT force OCR-only; hybrid runs via extract_pdf() mode
                    status_msg = "Extracting documents (hybrid: native + OCR with safety gate)..."
                else:
                    settings.use_ocr_for_all_documents = False
                    settings.ocr_enhancement_mode = "auto"
                    use_ocr = False
                    status_msg = "Extracting documents..."
                
                yield (
                    status_msg, None, None, None, 
                    gr.update(choices=[], value=None), None, None, None, None, 
                    None, None, None, None, None, None, ""
                )
                
                # Extract documents
                with track_time("extraction") as timing:
                    pages_a = extract_pdf(path_a, force_ocr=use_ocr, ocr_engine_priority=ocr_engine_priority_override)
                    pages_b = extract_pdf(path_b, force_ocr=use_ocr, ocr_engine_priority=ocr_engine_priority_override)
                extraction_time = timing.duration
                
                total_pages = max(len(pages_a), len(pages_b))
                status_msg = f"Extracted {len(pages_a)} and {len(pages_b)} pages. Comparing..."
                yield (
                    status_msg, None, None, None, 
                    gr.update(choices=[], value=None), None, None, None, None, 
                    None, None, None, None, None, None, ""
                )
                
                # Compare text
                with track_time("text_comparison") as timing:
                    comparator = TextComparator()
                    text_diffs = comparator.compare(pages_a, pages_b)
                text_time = timing.duration
                
                # Compare formatting
                with track_time("formatting_comparison") as timing:
                    formatting_diffs = compare_formatting(pages_a, pages_b)
                formatting_time = timing.duration
                
                # Compare tables
                with track_time("table_comparison") as timing:
                    table_diffs = compare_tables(pages_a, pages_b)
                table_time = timing.duration
                
                # Compare headers/footers
                with track_time("header_footer_comparison") as timing:
                    header_footer_diffs = compare_headers_footers(pages_a, pages_b)
                header_footer_time = timing.duration
                
                # Compare figure captions
                with track_time("figure_comparison") as timing:
                    figure_diffs = compare_figure_captions(pages_a, pages_b)
                figure_time = timing.duration
                
                # Combine and classify
                with track_time("classification") as timing:
                    # Fuse per-module diffs (pipeline parity + table-local absorption)
                    # before classification/merging to reduce noisy highlight clutter.
                    from comparison.diff_fusion import fuse_diff_lists

                    fused_diffs = fuse_diff_lists(
                        text_diffs,
                        formatting_diffs,
                        table_diffs,
                        header_footer_diffs,
                        figure_diffs,
                        module_names=[
                            "text",
                            "formatting",
                            "table",
                            "header_footer",
                            "figure",
                        ],
                    )

                    classified_diffs = classify_diffs(fused_diffs)
                    
                    # Merge nearby diffs to reduce clutter
                    classified_diffs = merge_nearby_diffs(classified_diffs)
                    
                    # Collapse into logical changes for metrics (keep surgical diffs for UI)
                    highlight_diffs, logical_changes = collapse_for_logical_changes(classified_diffs)
                classification_time = timing.duration
                
                # Create comparison result with logical changes for summary
                result = ComparisonResult(
                    doc1=str(path_a),
                    doc2=str(path_b),
                    pages=pages_a,  # Store pages from doc1
                    diffs=highlight_diffs,  # Surgical diffs for visualization
                    summary=get_diff_summary(logical_changes),  # Summary from logical changes
                )
                
                # Status shows both highlights and logical changes
                status_msg = f"Found {len(highlight_diffs)} highlights ({len(logical_changes)} logical changes). Rendering..."
                yield (
                    status_msg, None, None, None, 
                    gr.update(choices=[], value=None), result, None, None, None, 
                    None, None, None, None, None, None, ""
                )
                
                # Render pages with highlights (numpy arrays for Gallery)
                rendering_time = 0.0
                try:
                    with track_time("rendering") as timing:
                        rendered_a = render_pages(path_a, diffs=classified_diffs, doc_side="a")
                        rendered_b = render_pages(path_b, diffs=classified_diffs, doc_side="b")
                    rendering_time = timing.duration
                except Exception as exc:
                    logger.exception("Failed to render PDF pages: %s", exc)
                    # Fallback: render without diffs
                    try:
                        with track_time("rendering") as timing:
                            rendered_a = render_pages(path_a, diffs=None, doc_side="a")
                            rendered_b = render_pages(path_b, diffs=None, doc_side="b")
                        rendering_time = timing.duration
                    except Exception as exc2:
                        logger.exception("Failed to render PDF pages even without diffs: %s", exc2)
                        # Return empty galleries if rendering completely fails
                        rendered_a = []
                        rendered_b = []
                        rendering_time = 0.0
                
                # Overlay heatmap if requested
                if show_heat:
                    from comparison.visual_diff import generate_heatmap
                    from visualization.diff_renderer import overlay_heatmap
                    
                    heatmaps = generate_heatmap(path_a, path_b, dpi=144)
                    heatmap_dict = {page_num: heatmap for page_num, heatmap in heatmaps}
                    
                    # Overlay heatmaps on rendered pages
                    rendered_a_with_heatmap = []
                    rendered_b_with_heatmap = []
                    
                    for page_num, img_a in rendered_a:
                        if page_num in heatmap_dict:
                            img_a = overlay_heatmap(img_a, heatmap_dict[page_num], alpha=0.4)
                        rendered_a_with_heatmap.append((page_num, img_a))
                    
                    for page_num, img_b in rendered_b:
                        if page_num in heatmap_dict:
                            img_b = overlay_heatmap(img_b, heatmap_dict[page_num], alpha=0.4)
                        rendered_b_with_heatmap.append((page_num, img_b))
                    
                    rendered_a = rendered_a_with_heatmap
                    rendered_b = rendered_b_with_heatmap
                
                # Convert to format expected by Gradio Gallery (image, caption)
                # Ensure images are in correct format (RGB, uint8, contiguous)
                gallery_a_data = []
                gallery_b_data = []
                
                for page_num, img in rendered_a:
                    # Ensure image is in correct format for Gradio
                    if img is not None and img.size > 0:
                        # Make sure it's uint8 and has 3 channels (RGB)
                        if img.dtype != np.uint8:
                            img = img.astype(np.uint8)
                        if len(img.shape) == 2:  # Grayscale
                            img = np.stack([img] * 3, axis=2)
                        elif img.shape[2] == 4:  # RGBA
                            img = img[:, :, :3]
                        elif img.shape[2] != 3:  # Other format
                            logger.warning("Unexpected image shape: %s", img.shape)
                            continue
                        gallery_a_data.append((img, f"Page {page_num}"))
                
                for page_num, img in rendered_b:
                    # Ensure image is in correct format for Gradio
                    if img is not None and img.size > 0:
                        # Make sure it's uint8 and has 3 channels (RGB)
                        if img.dtype != np.uint8:
                            img = img.astype(np.uint8)
                        if len(img.shape) == 2:  # Grayscale
                            img = np.stack([img] * 3, axis=2)
                        elif img.shape[2] == 4:  # RGBA
                            img = img[:, :, :3]
                        elif img.shape[2] != 3:  # Other format
                            logger.warning("Unexpected image shape: %s", img.shape)
                            continue
                        gallery_b_data.append((img, f"Page {page_num}"))
                
                if not gallery_a_data:
                    logger.warning("No images rendered for Document A - PDF may be corrupted or rendering failed")
                    status_msg = f"Warning: Could not render Document A pages. {status_msg}"
                if not gallery_b_data:
                    logger.warning("No images rendered for Document B - PDF may be corrupted or rendering failed")
                    status_msg = f"Warning: Could not render Document B pages. {status_msg}"
                
                # Prepare diff list for display with visual indicators and severity
                diff_data = []
                diff_choices = []  # For dropdown
                low_confidence_count = 0
                
                for idx, diff in enumerate(classified_diffs):
                    # Calculate severity
                    severity = get_severity(diff.confidence)
                    if diff.confidence < 0.4:
                        low_confidence_count += 1
                    
                    # Format change type with visual indicator
                    change_type_icon = {
                        "content": "📝",
                        "formatting": "🎨",
                        "layout": "📐",
                        "visual": "👁️",
                    }.get(diff.change_type, "•")
                    
                    # Format diff type with color indicator
                    diff_type_indicator = {
                        "added": "➕",
                        "deleted": "➖",
                        "modified": "✏️",
                    }.get(diff.diff_type, "•")
                    
                    # Use enhanced description if available, otherwise generate fallback
                    desc = diff.metadata.get("description", "")
                    if not desc:
                        old_preview = diff.old_text[:60] if diff.old_text else ""
                        new_preview = diff.new_text[:60] if diff.new_text else ""
                        
                        if old_preview and new_preview:
                            desc = f"{old_preview} → {new_preview}"
                        elif old_preview:
                            desc = f"Removed: {old_preview}"
                        elif new_preview:
                            desc = f"Added: {new_preview}"
                        else:
                            # Formatting/layout change without text
                            formatting_type = diff.metadata.get("formatting_type", "change")
                            desc = f"{formatting_type.replace('_', ' ').title()} change"
                    
                    diff_data.append([
                        diff.page_num,
                        f"{diff_type_indicator} {diff.diff_type.title()}",
                        f"{change_type_icon} {diff.change_type.title()}",
                        severity,
                        desc,
                    ])
                    
                    # Add to dropdown choices
                    choice_label = f"Page {diff.page_num}: {change_type_icon} {diff.change_type.title()} - {desc[:50]}"
                    diff_choices.append((choice_label, idx))
                
                # Prepare alignment data for sync viewer.
                # Always compute a minimal identity page_map so scroll sync works even if
                # the user compares with sync OFF and toggles sync ON later.
                pages_a_count = len(pages_a or [])
                pages_b_count = len(pages_b or [])
                n = min(pages_a_count, pages_b_count)
                alignment_data = {
                    "pages_a": pages_a_count,
                    "pages_b": pages_b_count,
                    "page_map": {str(i): i for i in range(1, n + 1)},
                }
                
                # Calculate total time and per-page metrics
                total_time = extraction_time + text_time + formatting_time + table_time + \
                            header_footer_time + figure_time + classification_time + rendering_time
                time_per_page = total_time / total_pages if total_pages > 0 else 0.0
                
                # Count changes by type
                changes_by_type = {
                    "content": sum(1 for d in classified_diffs if d.change_type == "content"),
                    "formatting": sum(1 for d in classified_diffs if d.change_type == "formatting"),
                    "layout": sum(1 for d in classified_diffs if d.change_type == "layout"),
                    "visual": sum(1 for d in classified_diffs if d.change_type == "visual"),
                }
                
                # Get OCR engine info if OCR was used (OCR-only or hybrid).
                ocr_engine_info = ""
                try:
                    ocr_used = False
                    for p in (pages_a or []):
                        method = str((p.metadata or {}).get("extraction_method", ""))
                        if "ocr" in method.lower() or (p.metadata or {}).get("ocr_engine_used"):
                            ocr_used = True
                            break
                    if not ocr_used:
                        for p in (pages_b or []):
                            method = str((p.metadata or {}).get("extraction_method", ""))
                            if "ocr" in method.lower() or (p.metadata or {}).get("ocr_engine_used"):
                                ocr_used = True
                                break

                    if ocr_used:
                        md_a = (pages_a[0].metadata if pages_a else {}) or {}
                        md_b = (pages_b[0].metadata if pages_b else {}) or {}
                        engine_a = md_a.get("ocr_engine_used") or md_a.get("ocr_engine_selected")
                        engine_b = md_b.get("ocr_engine_used") or md_b.get("ocr_engine_selected")
                        attempts = md_a.get("ocr_attempts") or md_b.get("ocr_attempts")
                        skipped = md_a.get("ocr_preflight_skipped") or md_b.get("ocr_preflight_skipped") or {}
                        fallback = ""
                        if isinstance(attempts, list) and len(attempts) > 1:
                            chain = " → ".join(a.get("engine", "?") for a in attempts if isinstance(a, dict))
                            if chain:
                                fallback = f" (attempts: {chain})"

                        # If user selected DeepSeek but it was skipped during preflight, show why.
                        skip_note = ""
                        if (selected_ocr_engine or "").lower() == "deepseek" and isinstance(skipped, dict) and skipped.get("deepseek"):
                            reason = str(skipped.get("deepseek"))
                            if reason == "cuda_or_mps_unavailable":
                                reason_text = "CUDA/MPS unavailable"
                            elif reason == "dependency_missing":
                                reason_text = "dependencies missing"
                            else:
                                reason_text = reason
                            skip_note = f" (DeepSeek skipped: {reason_text})"

                        engine = engine_a or engine_b or "Unknown"
                        ocr_engine_info = f"**OCR Engine:** {engine}{fallback}{skip_note}\n"
                except Exception:
                    ocr_engine_info = "**OCR Engine:** Unknown\n"
                
                # Build performance metrics
                performance_metrics = {
                    "total_time": total_time,
                    "time_per_page": time_per_page,
                    "extraction_time": extraction_time,
                    "text_time": text_time,
                    "formatting_time": formatting_time,
                    "table_time": table_time,
                    "header_footer_time": header_footer_time,
                    "figure_time": figure_time,
                    "classification_time": classification_time,
                    "rendering_time": rendering_time,
                    "text_diffs_count": len(text_diffs),
                    "formatting_diffs_count": len(formatting_diffs),
                    "table_diffs_count": len(table_diffs),
                    "header_footer_diffs_count": len(header_footer_diffs),
                    "figure_diffs_count": len(figure_diffs),
                    "total_pages": total_pages,
                    "total_changes": len(classified_diffs),
                    "low_confidence_count": low_confidence_count,
                    "changes_by_type": changes_by_type,
                }
                
                # Phase 2: Compute OCR quality metrics if OCR was used
                ocr_quality_text = ""
                if use_ocr:
                    try:
                        from utils.ocr_quality_metrics import OCRQualityMetrics
                        ocr_metrics = OCRQualityMetrics()
                        ocr_metrics.engine_used = selected_ocr_engine or ""
                        ocr_metrics.analyze_diffs(classified_diffs, is_ocr=True)
                        
                        # Add to performance metrics
                        performance_metrics["ocr_quality"] = ocr_metrics.to_dict()
                        
                        # Format OCR quality section
                        precision_pct = ocr_metrics.precision_proxy * 100
                        phantom_pct = ocr_metrics.phantom_diff_ratio * 100
                        
                        # Color-code precision
                        if precision_pct >= 80:
                            precision_indicator = "🟢"
                        elif precision_pct >= 60:
                            precision_indicator = "🟡"
                        else:
                            precision_indicator = "🔴"
                        
                        ocr_quality_text = f"""
### 📊 OCR Quality Metrics
- **Precision Proxy:** {precision_indicator} {precision_pct:.1f}%
- **Phantom Diff Ratio:** {phantom_pct:.1f}%
- **Quality Score:** {ocr_metrics.quality_score:.0f}/100

### Severity Breakdown
- 🔴 Critical: {ocr_metrics.severity.critical}
- 🟠 High: {ocr_metrics.severity.high}
- 🟡 Medium: {ocr_metrics.severity.medium}
- 🟢 Low: {ocr_metrics.severity.low}
- ⚪ Phantom (filtered): {ocr_metrics.severity.none}

"""
                    except Exception as e:
                        logger.warning("Failed to compute OCR quality metrics: %s", e)
                        ocr_quality_text = ""

                def _fmt_seconds(seconds: float | None) -> str:
                    if seconds is None:
                        return "n/a"
                    # Avoid showing misleading 0.00s for fast stages.
                    if seconds < 0:
                        return "n/a"
                    if seconds < 0.001:
                        return "<1ms"
                    if seconds < 1.0:
                        return f"{seconds * 1000:.1f}ms"
                    return f"{seconds:.3f}s"
                
                # Format performance info
                perf_info_text = f"""
### Performance Summary
- **Total Processing Time:** {_fmt_seconds(total_time)}
- **Time per Page:** {_fmt_seconds(time_per_page)}
- **Pages Processed:** {total_pages}

### Timing Breakdown
- Extraction: {_fmt_seconds(extraction_time)}
- Text Comparison: {_fmt_seconds(text_time)} ({len(text_diffs)} raw diffs)
- Formatting Comparison: {_fmt_seconds(formatting_time)} ({len(formatting_diffs)} raw diffs)
- Table Comparison: {_fmt_seconds(table_time)} ({len(table_diffs)} raw diffs)
- Header/Footer Comparison: {_fmt_seconds(header_footer_time)} ({len(header_footer_diffs)} raw diffs)
- Figure Comparison: {_fmt_seconds(figure_time)} ({len(figure_diffs)} raw diffs)
- Classification: {_fmt_seconds(classification_time)} ({len(classified_diffs)} final diffs)
- Rendering: {_fmt_seconds(rendering_time)}

### Changes Detected
- **Total:** {len(classified_diffs)} changes
  - Content: {changes_by_type['content']}
  - Formatting: {changes_by_type['formatting']}
  - Layout: {changes_by_type['layout']}
  - Visual: {changes_by_type['visual']}
- **Low Confidence:** {low_confidence_count} (may need review)

{ocr_engine_info}{ocr_quality_text}
"""
                
                status_msg = f"Comparison complete! Found {len(classified_diffs)} differences."
                
                # Generate sync viewer HTML.
                # Important: use the image-based synced viewer (not iframes), so we can
                # reliably implement scroll sync + diff overlays.
                if use_sync_viewer_flag:
                    from visualization.custom_components.sync_pdf_viewer import (
                        _generate_pdf_html,
                    )

                    sync_viewer_a_html = _generate_pdf_html(
                        "a",
                        str(path_a),
                        None,
                        alignment_data=alignment_data or {},
                        diffs=classified_diffs,
                    )
                    sync_viewer_b_html = _generate_pdf_html(
                        "b",
                        str(path_b),
                        None,
                        alignment_data=alignment_data or {},
                        diffs=classified_diffs,
                    )
                else:
                    sync_viewer_a_html = ""
                    sync_viewer_b_html = ""
                
                yield (
                    status_msg,
                    gallery_a_data,
                    gallery_b_data,
                    diff_data,
                    gr.update(choices=diff_choices, value=None),  # diff_selector
                    result,
                    None,  # selected_diff_index
                    gallery_a_data,  # gallery_a_data_state
                    gallery_b_data,  # gallery_b_data_state
                    sync_viewer_a_html,  # sync_viewer_a
                    sync_viewer_b_html,  # sync_viewer_b
                    str(path_a),  # pdf_a_path_state
                    str(path_b),  # pdf_b_path_state
                    alignment_data,  # alignment_data_state
                    performance_metrics,  # performance_metrics_state
                    perf_info_text,  # performance_info
                )
                
            except Exception as exc:
                logger.exception("Comparison failed")
                error_msg = f"Error: {str(exc)}"
                yield (
                    error_msg, None, None, None, 
                    gr.update(choices=[], value=None), None, None, None, None, 
                    None, None, None, None, None, None, ""
                )
        
        def filter_diffs(
            result: ComparisonResult,
            show_content: bool,
            show_formatting: bool,
            show_layout: bool,
            show_visual: bool,
        ):
            """Filter diffs based on checkboxes."""
            if not result:
                return None, gr.update(choices=[], value=None)
            
            filtered = [
                diff for diff in result.diffs
                if (show_content and diff.change_type == "content") or
                   (show_formatting and diff.change_type == "formatting") or
                   (show_layout and diff.change_type == "layout") or
                   (show_visual and diff.change_type == "visual")
            ]
            
            # Format with visual indicators (same as in compare_documents)
            diff_data = []
            diff_choices = []
            
            # Create a mapping from filtered index to original index
            original_indices = []
            for idx, diff in enumerate(result.diffs):
                if diff in filtered:
                    original_indices.append(idx)
            
            for filtered_idx, diff in enumerate(filtered):
                # Calculate severity
                severity = get_severity(diff.confidence)
                
                change_type_icon = {
                    "content": "📝",
                    "formatting": "🎨",
                    "layout": "📐",
                    "visual": "👁️",
                }.get(diff.change_type, "•")
                
                diff_type_indicator = {
                    "added": "➕",
                    "deleted": "➖",
                    "modified": "✏️",
                }.get(diff.diff_type, "•")
                
                # Use enhanced description if available
                desc = diff.metadata.get("description", "")
                if not desc:
                    old_preview = diff.old_text[:60] if diff.old_text else ""
                    new_preview = diff.new_text[:60] if diff.new_text else ""
                    
                    if old_preview and new_preview:
                        desc = f"{old_preview} → {new_preview}"
                    elif old_preview:
                        desc = f"Removed: {old_preview}"
                    elif new_preview:
                        desc = f"Added: {new_preview}"
                    else:
                        formatting_type = diff.metadata.get("formatting_type", "change")
                        desc = f"{formatting_type.replace('_', ' ').title()} change"
                
                diff_data.append([
                    diff.page_num,
                    f"{diff_type_indicator} {diff.diff_type.title()}",
                    f"{change_type_icon} {diff.change_type.title()}",
                    severity,
                    desc,
                ])
                
                # Add to dropdown choices with original index
                original_idx = original_indices[filtered_idx]
                choice_label = f"Page {diff.page_num}: {change_type_icon} {diff.change_type.title()} - {desc[:50]}"
                diff_choices.append((choice_label, original_idx))
            
            return diff_data, gr.update(choices=diff_choices, value=None)
        
        def export_json_handler(result: ComparisonResult):
            """Export comparison result as JSON and return file for download."""
            if not result:
                return (
                    "No comparison result available.",
                    gr.update(visible=False, value=None),
                )
            
            try:
                # Create temp file with meaningful name
                doc1_name = Path(result.doc1).stem if result.doc1 else "doc1"
                doc2_name = Path(result.doc2).stem if result.doc2 else "doc2"
                filename = f"comparison_{doc1_name}_vs_{doc2_name}.json"
                
                temp_dir = tempfile.gettempdir()
                output_path = Path(temp_dir) / filename
                
                export_json(result, output_path)
                
                return (
                    f"✅ JSON exported: {filename}",
                    gr.update(visible=True, value=str(output_path)),
                )
            except Exception as exc:
                logger.exception("JSON export failed")
                return (
                    f"❌ Export failed: {str(exc)}",
                    gr.update(visible=False, value=None),
                )
        
        def export_pdf_handler(result: ComparisonResult):
            """Export comparison result as PDF and return file for download."""
            if not result:
                return (
                    "No comparison result available.",
                    gr.update(visible=False, value=None),
                )
            
            # Validate that result is actually a ComparisonResult object
            if not isinstance(result, ComparisonResult):
                logger.error("Expected ComparisonResult, got %s", type(result))
                return (
                    f"❌ Export failed: Invalid comparison result type: {type(result)}",
                    gr.update(visible=False, value=None),
                )
            
            # Ensure result has required attributes
            if not hasattr(result, 'doc1') or not hasattr(result, 'diffs'):
                logger.error("ComparisonResult missing required attributes")
                return (
                    "❌ Export failed: Invalid comparison result structure.",
                    gr.update(visible=False, value=None),
                )
            
            try:
                # Create temp file with meaningful name
                doc1_name = Path(result.doc1).stem if result.doc1 else "doc1"
                doc2_name = Path(result.doc2).stem if result.doc2 else "doc2"
                filename = f"comparison_report_{doc1_name}_vs_{doc2_name}.pdf"
                
                temp_dir = tempfile.gettempdir()
                output_path = Path(temp_dir) / filename
                
                export_pdf(result, output_path)
                
                return (
                    f"✅ PDF report exported: {filename}",
                    gr.update(visible=True, value=str(output_path)),
                )
            except Exception as exc:
                logger.exception("PDF export failed")
                return (
                    f"❌ Export failed: {str(exc)}",
                    gr.update(visible=False, value=None),
                )

        def _prepare_export_ui(kind: str):
            """Make the hidden download component visible before setting its value.

            Gradio can drop value updates when a component is hidden at the time
            the update is dispatched (seen with sync viewer + File downloads).
            We fix this by first toggling visibility, then chaining the export.
            """
            kind = (kind or "").lower().strip()
            if kind == "pdf":
                return (
                    "Preparing PDF export...",
                    gr.update(visible=True, value=None),
                )
            return (
                "Preparing JSON export...",
                gr.update(visible=True, value=None),
            )

        def prepare_json_export():
            """Prepare UI for JSON export (show hidden File output first)."""
            return _prepare_export_ui("json")

        def prepare_pdf_export():
            """Prepare UI for PDF export (show hidden File output first)."""
            return _prepare_export_ui("pdf")
        
        def handle_diff_selection(diff_index_str, result: ComparisonResult, gallery_a_data, gallery_b_data):
            """Handle diff selection - scroll to page and highlight bounding box."""
            if not result or diff_index_str is None or diff_index_str == "":
                return (
                    gr.update(),  # gallery_a
                    gr.update(),  # gallery_b
                    None,  # selected_diff_index
                )
            
            try:
                # diff_index_str is the index from the dropdown
                diff_index = int(diff_index_str)
                
                if diff_index < 0 or diff_index >= len(result.diffs):
                    return (gr.update(), gr.update(), None)
                
                selected_diff = result.diffs[diff_index]
                page_num = selected_diff.page_num
                
                # Calculate page index (0-based) for scrolling
                # Find the index of the page in the gallery
                page_index = None
                for i, (img, caption) in enumerate(gallery_a_data):
                    if f"Page {page_num}" in str(caption):
                        page_index = i
                        break
                
                if page_index is not None:
                    # Return galleries with selection to scroll to the page
                    # Gradio will automatically scroll to selected_index
                    return (
                        gr.update(value=gallery_a_data, selected_index=page_index),
                        gr.update(value=gallery_b_data, selected_index=page_index),
                        diff_index,
                    )
                else:
                    # Page not found, return unchanged
                    return (
                        gr.update(value=gallery_a_data),
                        gr.update(value=gallery_b_data),
                        diff_index,
                    )
            except (ValueError, IndexError, TypeError) as exc:
                logger.exception("Error handling diff selection: %s", exc)
                return (gr.update(), gr.update(), None)
        
        def navigate_previous_diff(
            current_index: int | None,
            result: ComparisonResult,
            gallery_a_data,
            gallery_b_data,
            show_content: bool,
            show_formatting: bool,
            show_layout: bool,
            show_visual: bool,
            use_sync: bool,
            pdf_a_path: str,
            pdf_b_path: str,
            alignment_data: dict,
        ):
            """Navigate to previous diff."""
            if not result or not result.diffs:
                return (gr.update(), gr.update(), None, gr.update(), gr.update())
            
            if not gallery_a_data or not gallery_b_data:
                return (gr.update(), gr.update(), None, gr.update(), gr.update())
            
            # Filter diffs based on current filter settings
            filtered = [
                diff for diff in result.diffs
                if (show_content and diff.change_type == "content") or
                   (show_formatting and diff.change_type == "formatting") or
                   (show_layout and diff.change_type == "layout") or
                   (show_visual and diff.change_type == "visual")
            ]
            
            if not filtered:
                return (gr.update(), gr.update(), None, gr.update(), gr.update())
            
            # Find current diff in filtered list
            if current_index is None or current_index < 0 or current_index >= len(result.diffs):
                # Start from last diff
                target_diff = filtered[-1]
            else:
                # Find current diff
                current_diff = result.diffs[current_index]
                if current_diff in filtered:
                    current_filtered_idx = filtered.index(current_diff)
                    if current_filtered_idx > 0:
                        target_diff = filtered[current_filtered_idx - 1]
                    else:
                        # Already at first, wrap to last
                        target_diff = filtered[-1]
                else:
                    # Current diff not in filtered, go to last
                    target_diff = filtered[-1]
            
            # Find original index
            target_index = result.diffs.index(target_diff)
            page_num = target_diff.page_num
            
            # Prepare sync viewer updates if in sync mode
            sync_a_update = gr.update()
            sync_b_update = gr.update()
            
            if use_sync and pdf_a_path and pdf_b_path:
                from visualization.custom_components.sync_pdf_viewer import (
                    _generate_pdf_html,
                )
                sync_a_update = gr.update(value=_generate_pdf_html("a", pdf_a_path, None, alignment_data=alignment_data or {}, page_num=page_num))
                sync_b_update = gr.update(value=_generate_pdf_html("b", pdf_b_path, None, alignment_data=alignment_data or {}, page_num=page_num))
            
            # Find page index in gallery
            page_index = None
            for i, (img, caption) in enumerate(gallery_a_data):
                if f"Page {page_num}" in str(caption):
                    page_index = i
                    break
            
            # Update gallery only if not in sync mode
            if page_index is not None and not use_sync:
                return (
                    gr.update(value=gallery_a_data, selected_index=page_index),
                    gr.update(value=gallery_b_data, selected_index=page_index),
                    target_index,
                    sync_a_update,
                    sync_b_update,
                )
            elif not use_sync:
                return (
                    gr.update(value=gallery_a_data),
                    gr.update(value=gallery_b_data),
                    target_index,
                    sync_a_update,
                    sync_b_update,
                )
            else:
                # In sync mode, don't update gallery
                return (
                    gr.update(),
                    gr.update(),
                    target_index,
                    sync_a_update,
                    sync_b_update,
                )
        
        def navigate_next_diff(
            current_index: int | None,
            result: ComparisonResult,
            gallery_a_data,
            gallery_b_data,
            show_content: bool,
            show_formatting: bool,
            show_layout: bool,
            show_visual: bool,
            use_sync: bool,
            pdf_a_path: str,
            pdf_b_path: str,
            alignment_data: dict,
        ):
            """Navigate to next diff."""
            if not result or not result.diffs:
                return (gr.update(), gr.update(), None, gr.update(), gr.update())
            
            if not gallery_a_data or not gallery_b_data:
                return (gr.update(), gr.update(), None, gr.update(), gr.update())
            
            # Filter diffs based on current filter settings
            filtered = [
                diff for diff in result.diffs
                if (show_content and diff.change_type == "content") or
                   (show_formatting and diff.change_type == "formatting") or
                   (show_layout and diff.change_type == "layout") or
                   (show_visual and diff.change_type == "visual")
            ]
            
            if not filtered:
                return (gr.update(), gr.update(), None, gr.update(), gr.update())
            
            # Find current diff in filtered list
            if current_index is None or current_index < 0 or current_index >= len(result.diffs):
                # Start from first diff
                target_diff = filtered[0]
            else:
                # Find current diff
                current_diff = result.diffs[current_index]
                if current_diff in filtered:
                    current_filtered_idx = filtered.index(current_diff)
                    if current_filtered_idx < len(filtered) - 1:
                        target_diff = filtered[current_filtered_idx + 1]
                    else:
                        # Already at last, wrap to first
                        target_diff = filtered[0]
                else:
                    # Current diff not in filtered, go to first
                    target_diff = filtered[0]
            
            # Find original index
            target_index = result.diffs.index(target_diff)
            page_num = target_diff.page_num
            
            # Prepare sync viewer updates if in sync mode
            sync_a_update = gr.update()
            sync_b_update = gr.update()
            
            if use_sync and pdf_a_path and pdf_b_path:
                from visualization.custom_components.sync_pdf_viewer import (
                    _generate_pdf_html,
                )
                sync_a_update = gr.update(value=_generate_pdf_html("a", pdf_a_path, None, alignment_data=alignment_data or {}, page_num=page_num))
                sync_b_update = gr.update(value=_generate_pdf_html("b", pdf_b_path, None, alignment_data=alignment_data or {}, page_num=page_num))
            
            # Find page index in gallery
            page_index = None
            for i, (img, caption) in enumerate(gallery_a_data):
                if f"Page {page_num}" in str(caption):
                    page_index = i
                    break
            
            # Update gallery only if not in sync mode
            if page_index is not None and not use_sync:
                return (
                    gr.update(value=gallery_a_data, selected_index=page_index),
                    gr.update(value=gallery_b_data, selected_index=page_index),
                    target_index,
                    sync_a_update,
                    sync_b_update,
                )
            elif not use_sync:
                return (
                    gr.update(value=gallery_a_data),
                    gr.update(value=gallery_b_data),
                    target_index,
                    sync_a_update,
                    sync_b_update,
                )
            else:
                # In sync mode, don't update gallery
                return (
                    gr.update(),
                    gr.update(),
                    target_index,
                    sync_a_update,
                    sync_b_update,
                )
        
        # Wire up events
        demo.load(fn=None, inputs=None, outputs=None, js=GLOBAL_CLIENT_LOAD_JS, queue=False)

        use_sync_viewer.change(
            toggle_viewer_visibility,
            inputs=[use_sync_viewer],
            outputs=[gallery_row, sync_viewer_container],
        ).then(
            render_sync_viewer_content,
            inputs=[use_sync_viewer, comparison_result, pdf_a_path_state, pdf_b_path_state, alignment_data_state],
            outputs=[sync_viewer_a, sync_viewer_b],
        )
        
        # OCR engine dropdown is always visible now (DeepSeek default).
        
        compare_btn.click(
            compare_documents,
            inputs=[
                doc1, doc2, show_heatmap, scanned_document_mode, 
                force_ocr, sensitivity_threshold, use_sync_viewer, ocr_engine,
                text_model, layout_model,
            ],
            outputs=[
                status,
                gallery_a,
                gallery_b,
                diff_list,
                diff_selector,
                comparison_result,
                selected_diff_index,
                gallery_a_data_state,
                gallery_b_data_state,
                sync_viewer_a,
                sync_viewer_b,
                pdf_a_path_state,
                pdf_b_path_state,
                alignment_data_state,
                performance_metrics_state,
                performance_info,
            ],
        )
        
        # Handle diff selector dropdown change
        diff_selector.change(
            handle_diff_selection,
            inputs=[diff_selector, comparison_result, gallery_a_data_state, gallery_b_data_state],
            outputs=[gallery_a, gallery_b, selected_diff_index],
        )
        
        # Navigation buttons
        prev_diff_btn.click(
            navigate_previous_diff,
            inputs=[
                selected_diff_index, comparison_result, gallery_a_data_state, 
                gallery_b_data_state, filter_content, filter_formatting, 
                filter_layout, filter_visual,
                use_sync_viewer, pdf_a_path_state, pdf_b_path_state, alignment_data_state
            ],
            outputs=[gallery_a, gallery_b, selected_diff_index, sync_viewer_a, sync_viewer_b],
        )
        
        next_diff_btn.click(
            navigate_next_diff,
            inputs=[
                selected_diff_index, comparison_result, gallery_a_data_state, 
                gallery_b_data_state, filter_content, filter_formatting, 
                filter_layout, filter_visual,
                use_sync_viewer, pdf_a_path_state, pdf_b_path_state, alignment_data_state
            ],
            outputs=[gallery_a, gallery_b, selected_diff_index, sync_viewer_a, sync_viewer_b],
        )
        
        # Wire up filter events
        filter_content.change(
            filter_diffs,
            inputs=[comparison_result, filter_content, filter_formatting, filter_layout, filter_visual],
            outputs=[diff_list, diff_selector],
        )
        
        filter_formatting.change(
            filter_diffs,
            inputs=[comparison_result, filter_content, filter_formatting, filter_layout, filter_visual],
            outputs=[diff_list, diff_selector],
        )
        
        filter_layout.change(
            filter_diffs,
            inputs=[comparison_result, filter_content, filter_formatting, filter_layout, filter_visual],
            outputs=[diff_list, diff_selector],
        )
        
        filter_visual.change(
            filter_diffs,
            inputs=[comparison_result, filter_content, filter_formatting, filter_layout, filter_visual],
            outputs=[diff_list, diff_selector],
        )
        
        # Zoom controls are handled entirely in the browser via the JS listeners
        # injected earlier in the UI. We intentionally do NOT register backend
        # `.change()` handlers for the sliders, because Gradio's slider preprocess
        # can crash if the frontend ever sends `null` for a slider value.
        
        export_json_btn.click(
            prepare_json_export,
            inputs=[],
            outputs=[status, json_download],
            queue=False,
        ).then(
            export_json_handler,
            inputs=[comparison_result],
            outputs=[status, json_download],
        )
        
        export_pdf_btn.click(
            prepare_pdf_export,
            inputs=[],
            outputs=[status, pdf_download],
            queue=False,
        ).then(
            export_pdf_handler,
            inputs=[comparison_result],
            outputs=[status, pdf_download],
        )
    
    return demo
