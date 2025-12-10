"""Custom Gradio components for synchronized PDF viewing."""
from __future__ import annotations

from visualization.custom_components.sync_pdf_viewer import (
    SyncPDFViewer,
    _generate_diff_list,
    _generate_pdf_html,
    create_sync_pdf_viewer,
    update_sync_viewer,
)

__all__ = [
    "SyncPDFViewer",
    "create_sync_pdf_viewer",
    "update_sync_viewer",
    "_generate_diff_list",
    "_generate_pdf_html",
]

