"""Synchronized PDF viewer component with semantic scrolling.

Note:
PDF iframes cannot be reliably scroll-synchronized or overlaid with bbox highlights
because the browser PDF viewer is not scriptable in a cross-browser way.

This module implements a robust approach by rendering PDF pages to images and
building a scrollable HTML viewer where we control scrolling and diff overlays.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

import numpy as np
from PIL import Image

from comparison.models import ComparisonResult, Diff
from utils.logging import logger
from visualization.pdf_viewer import render_pages


def create_sync_pdf_viewer(
    pdf_a_path: Optional[str] = None,
    pdf_b_path: Optional[str] = None,
    comparison_result: Optional[ComparisonResult] = None,
    alignment_data: Optional[Dict] = None,
) -> Tuple[gr.HTML, gr.HTML, gr.Dataframe]:
    """
    Create synchronized PDF viewer components.
    
    Returns:
        Tuple of (pdf_viewer_a, pdf_viewer_b, diff_list) components
    """
    # Generate HTML with scroll sync script included
    sync_script = _add_scroll_sync_script(alignment_data or {})
    diffs = comparison_result.diffs if comparison_result else None
    html_a = _generate_pdf_html("a", pdf_a_path, sync_script, alignment_data=alignment_data or {}, diffs=diffs)
    html_b = _generate_pdf_html("b", pdf_b_path, sync_script, alignment_data=alignment_data or {}, diffs=diffs)
    
    pdf_viewer_a = gr.HTML(
        value=html_a,
        elem_id="pdf-viewer-a",
    )
    pdf_viewer_b = gr.HTML(
        value=html_b,
        elem_id="pdf-viewer-b",
    )
    diff_list = gr.Dataframe(
        label="Detected Changes",
        headers=["Page", "Type", "Description"],
        interactive=False,
        value=_generate_diff_list(comparison_result),
    )
    
    return pdf_viewer_a, pdf_viewer_b, diff_list


class SyncPDFViewer:
    """
    Custom Gradio component for synchronized PDF viewing with semantic scrolling.
    
    This component displays two PDFs side-by-side with:
    - Semantic scroll synchronization (based on aligned paragraphs)
    - Visual diff overlays (colored bounding boxes)
    - Jump-to-change navigation
    """
    
    """Placeholder class for backward compatibility."""
    pass


def _generate_pdf_html(
    viewer_id: str, 
    pdf_path: Optional[str], 
    sync_script: Optional[str] = None,
    alignment_data: Optional[Dict] = None,
    page_num: Optional[int] = None,
    diffs: Optional[List[Diff]] = None,
    dpi: int = 144,
    scale_factor: float = 2.0,
) -> str:
    """Generate an image-based PDF viewer that supports scroll sync + diff overlays."""
    if not pdf_path:
        return "<div style='color:#94a3b8;padding:20px;text-align:center;'>No PDF loaded</div>"

    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        logger.error("PDF file not found: %s", pdf_path)
        return (
            "<div style='color:#ef4444;padding:20px;background:rgba(239,68,68,0.1);"
            "border-radius:8px;border:1px solid rgba(239,68,68,0.3);'>"
            f"<strong>⚠️ PDF file not found:</strong> {pdf_path}"
            "</div>"
        )

    try:
        rendered = render_pages(
            pdf_path_obj,
            dpi=dpi,
            diffs=diffs,
            scale_factor=scale_factor,
            doc_side=viewer_id,
        )
    except Exception as exc:
        logger.exception("Failed to render PDF %s: %s", pdf_path, exc)
        return (
            "<div style='color:#ef4444;padding:20px;background:rgba(239,68,68,0.1);"
            "border-radius:8px;border:1px solid rgba(239,68,68,0.3);'>"
            f"<strong>⚠️ Render error:</strong> {exc}"
            "</div>"
        )

    def _img_to_data_uri(img: np.ndarray) -> str:
        pil = Image.fromarray(img.astype(np.uint8), mode="RGB")
        import io
        import base64

        buf = io.BytesIO()
        pil.save(buf, format="PNG", optimize=True)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    pages_html: List[str] = []
    for page_number, page_img in rendered:
        data_uri = _img_to_data_uri(page_img)
        pages_html.append(
            "<div class='sync-page' data-page='" + str(page_number) + "'>"
            "  <div class='sync-page-inner'>"
            "    <img class='sync-page-img' src='" + data_uri + "' alt='Page " + str(page_number) + "' />"
            "    <div class='sync-page-label'>Page " + str(page_number) + "</div>"
            "  </div>"
            "</div>"
        )

    # Inline styles for this viewer (kept local so it works even if global CSS changes)
    style = """
    <style>
      .sync-img-viewer { height: 800px; overflow: auto; background: #0f172a; border-radius: 12px; border: 1px solid rgba(148,163,184,0.2); padding: 12px; }
      .sync-page { margin: 0 0 16px 0; }
      .sync-page-inner { position: relative; display: block; }
      .sync-page-img { width: 100%; height: auto; display: block; border-radius: 8px; }
      .sync-page-label { position: absolute; top: 10px; left: 10px; background: rgba(2,6,23,0.6); color: #e2e8f0; font-size: 12px; padding: 4px 8px; border-radius: 8px; border: 1px solid rgba(148,163,184,0.25); }
            .sync-bbox-overlay { position: absolute; border: 2px solid rgba(255,215,0,0.8); background: rgba(255,215,0,0.20); border-radius: 4px; min-width: 4px; min-height: 4px; pointer-events: none; box-shadow: 0 0 10px rgba(255,215,0,0.35); }
    </style>
    """

    jump_script = ""
    if page_num:
        jump_script = f"""
        <script>
          setTimeout(function() {{
            try {{ window.pdfViewer{viewer_id.upper()}?.scrollToPage?.({int(page_num)}); }} catch(e) {{}}
          }}, 250);
        </script>
        """

    # Embed alignment data into DOM attributes so sync can be initialized even if
    # <script> tags inside gr.HTML don't execute in some Gradio versions.
    try:
        alignment_attr = json.dumps(alignment_data or {}, separators=(",", ":"))
    except Exception:
        alignment_attr = "{}"
    # Escape quotes for safe embedding in HTML attribute
    alignment_attr = alignment_attr.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")

    html = (
        style
        + f"<div id='sync-img-viewer-{viewer_id}' class='sync-img-viewer' data-alignment=\"{alignment_attr}\">"
        + "\n".join(pages_html)
        + "</div>"
        + (sync_script or "")
        + jump_script
    )
    return html



def _generate_diff_list(comparison_result: Optional[ComparisonResult]) -> List[List[str]]:
    """Generate list of diffs for navigation."""
    if not comparison_result or not comparison_result.diffs:
        return []
    
    diff_data = []
    for diff in comparison_result.diffs:
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
        
        diff_data.append([
            str(diff.page_num),
            f"{diff_type_indicator} {diff.diff_type.title()}",
            f"{change_type_icon} {diff.change_type.title()} - {desc[:50]}",
        ])
    
    return diff_data


def _add_scroll_sync_script(alignment_data: Dict):
    """Add JavaScript for scroll synchronization and diff highlighting.

    Works with the image-based viewers generated by `_generate_pdf_html`.
    """
    alignment_json = json.dumps(alignment_data)
    
    # Generate enhanced scroll sync script that works with premium PDF viewers
    script = f"""
    <script>
    (function() {{
        const alignmentData = {alignment_json};
        
        // Diff type color mapping (matching React example)
        const DIFF_COLORS = {{
            'added': {{
                border: 'rgba(0, 255, 0, 0.7)',
                background: 'rgba(0, 255, 0, 0.2)',
                glow: 'rgba(0, 255, 0, 0.5)'
            }},
            'deleted': {{
                border: 'rgba(255, 0, 0, 0.7)',
                background: 'rgba(255, 0, 0, 0.2)',
                glow: 'rgba(255, 0, 0, 0.5)'
            }},
            'modified': {{
                border: 'rgba(255, 215, 0, 0.7)',
                background: 'rgba(255, 215, 0, 0.2)',
                glow: 'rgba(255, 215, 0, 0.5)'
            }}
        }};
        
        function buildViewer(id) {{
            const container = document.getElementById(`sync-img-viewer-${{id}}`);
            if (!container) return null;

            function getPageEl(pageNum) {{
                return container.querySelector(`.sync-page[data-page="${{pageNum}}"]`);
            }}

            function clearHighlights() {{
                container.querySelectorAll('.sync-bbox-overlay').forEach(el => el.remove());
            }}

            function scrollToPage(pageNum) {{
                const pageEl = getPageEl(pageNum);
                if (pageEl) pageEl.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
            }}

            function highlightBBox(pageNum, bbox, diffType) {{
                if (!bbox) return null;
                const pageEl = getPageEl(pageNum);
                if (!pageEl) return null;

                const inner = pageEl.querySelector('.sync-page-inner');
                if (!inner) return null;

                const colors = DIFF_COLORS[diffType || 'modified'] || DIFF_COLORS['modified'];

                const x = Number(bbox.x ?? bbox.left ?? 0);
                const y = Number(bbox.y ?? bbox.top ?? 0);
                const w = Number(bbox.width ?? 0);
                const h = Number(bbox.height ?? 0);

                const overlay = document.createElement('div');
                overlay.className = 'sync-bbox-overlay';

                // Use pixel units to avoid tiny boxes rounding to 0% at high zoom.
                const rect = inner.getBoundingClientRect();
                const cx = Math.max(0, Math.min(1, x)) * rect.width;
                const cy = Math.max(0, Math.min(1, y)) * rect.height;
                const cw = Math.max(0, Math.min(1, w)) * rect.width;
                const ch = Math.max(0, Math.min(1, h)) * rect.height;
                overlay.style.left = `${{cx}}px`;
                overlay.style.top = `${{cy}}px`;
                overlay.style.width = `${{cw}}px`;
                overlay.style.height = `${{ch}}px`;
                overlay.style.borderColor = colors.border;
                overlay.style.backgroundColor = colors.background;
                overlay.style.boxShadow = `0 0 10px ${{colors.glow}}`;
                inner.appendChild(overlay);
                return overlay;
            }}

            return {{ container, scrollToPage, highlightBBox, clearHighlights }};
        }}

        // Global scroll sync initialization function
        window.initScrollSync = function() {{
            const viewerA = buildViewer('a');
            const viewerB = buildViewer('b');
            if (!viewerA || !viewerB) return;

            window.pdfViewerA = viewerA;
            window.pdfViewerB = viewerB;

            console.log('📐 Initializing image-based scroll synchronization');

            // Page-by-page sync: when you scroll to a new page in one viewer,
            // the other viewer scrolls to the corresponding page (block sync).
            const pageMap = (alignmentData && (alignmentData.page_map || alignmentData.alignment_map)) || {{}};

            function getActivePageNum(viewer) {{
                const pages = viewer.container.querySelectorAll('.sync-page');
                if (!pages || pages.length === 0) return null;

                const y = viewer.container.scrollTop;
                const bias = 24; // tolerate minor offsets
                let active = pages[0];

                // pages are appended in order, so we can walk until we pass current scroll
                for (let i = 0; i < pages.length; i++) {{
                    const p = pages[i];
                    if (p.offsetTop <= y + bias) active = p;
                    else break;
                }}

                const pageStr = active?.dataset?.page;
                const pageNum = pageStr ? Number(pageStr) : null;
                return Number.isFinite(pageNum) ? pageNum : null;
            }}

            function mapPage(sourceId, pageNum) {{
                if (!pageNum) return pageNum;
                try {{
                    const key = String(pageNum);
                    const mapped = pageMap?.[key];
                    if (mapped) return Number(mapped) || pageNum;
                }} catch (e) {{}}
                return pageNum;
            }}

            let isSyncing = false;
            let lastPageA = null;
            let lastPageB = null;

            function syncByPage(source, target, sourceId) {{
                if (isSyncing) return;

                const sourcePage = getActivePageNum(source);
                if (!sourcePage) return;

                if (sourceId === 'a') {{
                    if (sourcePage === lastPageA) return;
                    lastPageA = sourcePage;
                }} else {{
                    if (sourcePage === lastPageB) return;
                    lastPageB = sourcePage;
                }}

                const targetPage = mapPage(sourceId, sourcePage);
                const targetActive = getActivePageNum(target);
                if (targetActive === targetPage) return;

                isSyncing = true;
                target.scrollToPage(targetPage);
                setTimeout(() => {{ isSyncing = false; }}, 200);
            }}

            // Lightweight debounce so continuous scrolling doesn't spam scrollIntoView.
            let tA = null;
            let tB = null;
            viewerA.container.addEventListener('scroll', () => {{
                if (tA) clearTimeout(tA);
                tA = setTimeout(() => syncByPage(viewerA, viewerB, 'a'), 60);
            }}, {{ passive: true }});
            viewerB.container.addEventListener('scroll', () => {{
                if (tB) clearTimeout(tB);
                tB = setTimeout(() => syncByPage(viewerB, viewerA, 'b'), 60);
            }}, {{ passive: true }});

            console.log('✅ Scroll synchronization active');
        }};
        
        // Function to highlight diffs on PDF viewers (called from diff selector)
        window.highlightDiffOnViewers = function(diff) {{
            const viewerA = window.pdfViewerA;
            const viewerB = window.pdfViewerB;
            if (!viewerA || !viewerB || !diff) return;

            const pageNumA = diff.page_num || diff.page;
            const pageNumB = diff.page_num_b || diff.page_num || diff.page;
            const bboxA = diff.bbox;
            const bboxB = diff.bbox_b || diff.bbox;
            const diffType = diff.diff_type || 'modified';
            if (!pageNumA || !bboxA) return;

            viewerA.clearHighlights();
            viewerB.clearHighlights();

            viewerA.highlightBBox(pageNumA, bboxA, diffType);
            if (pageNumB && bboxB) {{
                viewerB.highlightBBox(pageNumB, bboxB, diffType);
            }}
            viewerA.scrollToPage(pageNumA);
            if (pageNumB) {{
                viewerB.scrollToPage(pageNumB);
            }}
        }};
        
        // Try to initialize immediately if viewers are already loaded
        if (document.readyState === 'complete') {{
            window.initScrollSync();
        }} else {{
            window.addEventListener('load', function() {{
                setTimeout(window.initScrollSync, 1000);
            }});
        }}
    }})();
    </script>
    """
    
    return script


def update_sync_viewer(
    pdf_a_path: Optional[str],
    pdf_b_path: Optional[str],
    comparison_result: Optional[ComparisonResult],
    alignment_data: Optional[Dict],
) -> Tuple[gr.update, gr.update, gr.update]:
    """Update the synchronized viewer with new data."""
    sync_script = _add_scroll_sync_script(alignment_data or {})
    diffs = comparison_result.diffs if comparison_result else None
    return (
        gr.update(value=_generate_pdf_html("a", pdf_a_path, sync_script, alignment_data=alignment_data or {}, diffs=diffs)),
        gr.update(value=_generate_pdf_html("b", pdf_b_path, sync_script, alignment_data=alignment_data or {}, diffs=diffs)),
        gr.update(value=_generate_diff_list(comparison_result)),
    )

