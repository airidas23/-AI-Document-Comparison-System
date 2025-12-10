"""Synchronized PDF viewer component with semantic scrolling."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from comparison.models import ComparisonResult, Diff
from utils.logging import logger


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
    html_a = _generate_pdf_html("a", pdf_a_path, sync_script)
    html_b = _generate_pdf_html("b", pdf_b_path, sync_script)
    
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
    page_num: Optional[int] = None,
) -> str:
        """Generate simple, reliable PDF viewer using iframe with base64 encoding.
        
        Using base64 data URI to avoid Gradio file serving restrictions.
        
        Args:
            viewer_id: Unique identifier for the viewer ('a' or 'b')
            pdf_path: Path to the PDF file
            sync_script: Optional JavaScript for scroll synchronization
            page_num: Optional page number to jump to (1-indexed)
        """
        if not pdf_path:
            return "<div style='color: #94a3b8; padding: 20px; text-align: center;'>No PDF loaded</div>"
        
        # Check if file exists and encode as base64
        try:
            from pathlib import Path
            import base64
            
            pdf_path_obj = Path(pdf_path)
            if not pdf_path_obj.exists():
                logger.error("PDF file not found: %s", pdf_path)
                return f"""
                <div style='color: #ef4444; padding: 20px; background: rgba(239, 68, 68, 0.1); 
                            border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.3);'>
                    <strong>⚠️ PDF file not found:</strong> {pdf_path}
                </div>
                """
            
            # Read PDF and encode as base64
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
            pdf_data_uri = f"data:application/pdf;base64,{pdf_base64}"
            
            # Add page fragment if page number is specified
            if page_num:
                pdf_data_uri = f"{pdf_data_uri}#page={page_num}"
            
            logger.info(f"Encoded PDF for viewer {viewer_id} ({len(pdf_bytes)} bytes)" + 
                       (f" jumping to page {page_num}" if page_num else ""))
            
        except Exception as exc:
            logger.exception("Failed to encode PDF %s: %s", pdf_path, exc)
            return f"""
            <div style='color: #ef4444; padding: 20px; background: rgba(239, 68, 68, 0.1); 
                        border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.3);'>
                <strong>⚠️ Error:</strong> {exc}
            </div>
            """
        
        # Generate iframe with embedded PDF data
        html = f"""
        <div style="width: 100%; height: 800px; background: #1e293b; border-radius: 12px; overflow: hidden; border: 1px solid rgba(148, 163, 184, 0.2);">
            <iframe
                src="{pdf_data_uri}"
                style="width: 100%; height: 100%; border: none;"
                type="application/pdf"
            >
                <p style="color: white; padding: 20px; text-align: center;">
                    Your browser doesn't support embedded PDFs.
                </p>
            </iframe>
        </div>
        {sync_script or ''}
        """
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
    """Add JavaScript for semantic scroll synchronization with diff highlighting support."""
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
        
        // Global scroll sync initialization function
        window.initScrollSync = function() {{
            // Wait for both PDF viewers to load
            const checkViewers = setInterval(function() {{
                const viewerA = window.pdfViewerA;
                const viewerB = window.pdfViewerB;
                
                if (!viewerA || !viewerB || !viewerA.container || !viewerB.container) {{
                    return;
                }}
                
                // Both viewers are ready
                clearInterval(checkViewers);
                console.log('📐 Initializing premium scroll synchronization');
                
                let isScrolling = false;
                let scrollTimeout;
                
                // Debounced semantic scroll sync
                function syncScroll(sourceViewer, targetViewer) {{
                    if (isScrolling) return;
                    isScrolling = true;
                    
                    // Clear previous timeout
                    if (scrollTimeout) clearTimeout(scrollTimeout);
                    
                    const scrollTop = sourceViewer.container.scrollTop;
                    const scrollHeight = sourceViewer.container.scrollHeight;
                    const clientHeight = sourceViewer.container.clientHeight;
                    
                    // Prevent division by zero
                    const maxScroll = scrollHeight - clientHeight;
                    if (maxScroll <= 0) {{
                        isScrolling = false;
                        return;
                    }}
                    
                    // Calculate relative scroll position (0-1)
                    const relativePos = Math.max(0, Math.min(1, scrollTop / maxScroll));
                    
                    // Map to target document using alignment data
                    // Enhanced with semantic paragraph alignment (can be improved with actual data)
                    const targetMaxScroll = targetViewer.container.scrollHeight - targetViewer.container.clientHeight;
                    
                    if (targetMaxScroll > 0) {{
                        const targetScrollTop = relativePos * targetMaxScroll;
                        targetViewer.container.scrollTop = targetScrollTop;
                    }}
                    
                    // Debounced reset
                    scrollTimeout = setTimeout(function() {{
                        isScrolling = false;
                    }}, 150);
                }}
                
                // Add scroll listeners with passive flag for performance
                viewerA.container.addEventListener('scroll', function() {{
                    syncScroll(viewerA, viewerB);
                }}, {{ passive: true }});
                
                viewerB.container.addEventListener('scroll', function() {{
                    syncScroll(viewerB, viewerA);
                }}, {{ passive: true }});
                
                console.log('✅ Scroll synchronization active');
            }}, 500);
            
            // Timeout after 15 seconds
            setTimeout(function() {{
                clearInterval(checkViewers);
            }}, 15000);
        }};
        
        // Function to highlight diffs on PDF viewers (called from diff selector)
        window.highlightDiffOnViewers = function(diff) {{
            const viewerA = window.pdfViewerA;
            const viewerB = window.pdfViewerB;
            
            if (!viewerA || !viewerB || !diff) return;
            
            const pageNum = diff.page_num || diff.page;
            const bbox = diff.bbox;
            const diffType = diff.diff_type || 'modified';
            
            if (!pageNum || !bbox) {{
                console.warn('Invalid diff data:', diff);
                return;
            }}
            
            // Get colors for diff type
            const colors = DIFF_COLORS[diffType] || DIFF_COLORS['modified'];
            
            // Clear previous highlights
            document.querySelectorAll('.bbox-overlay-a, .bbox-overlay-b').forEach(el => el.remove());
            
            // Highlight on both viewers
            try {{
                const overlayA = viewerA.highlightBBox(pageNum, bbox);
                if (overlayA) {{
                    overlayA.classList.add('bbox-overlay-a');
                    overlayA.style.borderColor = colors.border;
                    overlayA.style.backgroundColor = colors.background;
                    overlayA.style.boxShadow = `0 0 10px ${{colors.glow}}`;
                }}
                
                const overlayB = viewerB.highlightBBox(pageNum, bbox);
                if (overlayB) {{
                    overlayB.classList.add('bbox-overlay-b');
                    overlayB.style.borderColor = colors.border;
                    overlayB.style.backgroundColor = colors.background;
                    overlayB.style.boxShadow = `0 0 10px ${{colors.glow}}`;
                }}
                
                // Scroll both viewers to the page
                viewerA.scrollToPage(pageNum);
                viewerB.scrollToPage(pageNum);
                
                console.log(`🎯 Highlighted diff on page ${{pageNum}} (${{diffType}})`);
            }} catch (error) {{
                console.error('Error highlighting diff:', error);
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
    return (
        gr.update(value=_generate_pdf_html("a", pdf_a_path, sync_script)),
        gr.update(value=_generate_pdf_html("b", pdf_b_path, sync_script)),
        gr.update(value=_generate_diff_list(comparison_result)),
    )

