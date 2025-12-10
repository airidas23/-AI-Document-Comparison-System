"""Gradio interface components."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Tuple

import gradio as gr
import numpy as np

from comparison.diff_classifier import classify_diffs, get_diff_summary
from comparison.figure_comparison import compare_figure_captions
from comparison.formatting_comparison import compare_formatting
from comparison.models import ComparisonResult
from comparison.table_comparison import compare_tables
from comparison.text_comparison import TextComparator
from comparison.visual_diff import generate_heatmap_bytes
from config.settings import settings
from export.json_exporter import export_json
from export.pdf_exporter import export_pdf
from extraction import extract_pdf
from extraction.header_footer_detector import compare_headers_footers
from extraction.ocr_router import is_cuda_available
from utils.logging import logger
from utils.performance import clear_timings, get_timings, track_time
from utils.validation import validate_pdf_path
from visualization.custom_components.sync_pdf_viewer import (
    create_sync_pdf_viewer,
    update_sync_viewer,
)
from visualization.pdf_viewer import render_page_pair, render_pages


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
            value=settings.text_similarity_threshold,
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
            label="Force OCR for All Documents",
            value=False,
            info="Use OCR even if PDF text is available. Useful if PDF text extraction is unreliable.",
        )
        
        show_heatmap = gr.Checkbox(
            label="Show Heatmap Overlay",
            value=False,
            info="Overlay a translucent heatmap showing all differences on PDF pages",
        )
        
        # OCR Engine Selection (only visible when Force OCR is enabled)
        ocr_engine_options = ["paddle", "deepseek", "tesseract"]
        ocr_engine = gr.Dropdown(
            choices=ocr_engine_options,
            value=settings.ocr_engine,
            label="OCR Engine",
            info="Select OCR engine. DeepSeek requires GPU/MPS. PaddleOCR is fast/accurate. Tesseract is legacy.",
            visible=False,  # Initially hidden, shown only when Force OCR is enabled
        )
    
    return params_accordion, sensitivity_threshold, scanned_document_mode, force_ocr, show_heatmap, ocr_engine


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
        // ZOOM CONTROLS
        // Apply zoom to gallery images (per-gallery, resilient to Gradio DOM changes)
        // =====================================================
        (function() {
            function applyZoomToGallery(galleryId, zoomLevel) {
                const galleryRoot = document.getElementById(galleryId);
                if (!galleryRoot) return;
                const images = galleryRoot.querySelectorAll('img');
                images.forEach(img => {
                    img.style.transform = `scale(${zoomLevel / 100})`;
                    img.style.transformOrigin = 'top left';
                    img.style.transition = 'transform 0.15s ease';
                });
            }

            function setupZoomControl(sliderContainerId, galleryId) {
                // Gradio puts elem_id on a wrapper; the range input is inside it
                const container = document.getElementById(sliderContainerId);
                if (!container) return;
                const slider = container.querySelector('input[type="range"]');
                if (!slider) return;

                const applyCurrentZoom = () => {
                    const zoomLevel = parseFloat(slider.value) || 100;
                    applyZoomToGallery(galleryId, zoomLevel);
                };

                slider.addEventListener('input', applyCurrentZoom);
                applyCurrentZoom();

                // Re-apply zoom when gallery content updates (e.g., new pages rendered)
                const galleryRoot = document.getElementById(galleryId);
                if (galleryRoot) {
                    const observer = new MutationObserver(() => applyCurrentZoom());
                    observer.observe(galleryRoot, { childList: true, subtree: true });
                }
            }

            function initZoomControls() {
                setupZoomControl('zoom-a-slider', 'gallery-a');
                setupZoomControl('zoom-b-slider', 'gallery-b');
            }

            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', initZoomControls);
            } else {
                initZoomControls();
            }

            // Also re-run after Gradio DOM updates (debounced)
            let setupTimeout;
            const observer = new MutationObserver(function() {
                clearTimeout(setupTimeout);
                setupTimeout = setTimeout(initZoomControls, 300);
            });
            observer.observe(document.body, { childList: true, subtree: true });

            console.log('✨ Zoom controls initialized');
        })();
        </script>
        """, visible=False)
        gr.Markdown("# 🔍 AI Document Comparison System")
        gr.Markdown("""
        Upload two PDF documents to compare and visualize differences with AI-powered analysis.
        
        **Keyboard Shortcuts:** Press `j` for next diff | `k` for previous diff | `Esc` to clear highlights
        """)
        
        with gr.Row():
            doc1, doc2 = build_upload_row()
        
        # Parameters Panel (collapsible)
        params_accordion, sensitivity_threshold, scanned_document_mode, force_ocr, show_heatmap, ocr_engine = build_parameters_panel()
        
        with gr.Row():
            compare_btn = gr.Button("Compare Documents", variant="primary")
            use_sync_viewer = gr.Checkbox(
                label="Use Synchronized Viewer",
                value=False,
                info="Enable advanced synchronized PDF viewer with semantic scrolling"
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
                    zoom_a = gr.Slider(
                        minimum=50,
                        maximum=200,
                        value=100,
                        step=10,
                        label="Zoom (%)",
                        elem_id="zoom-a-slider",
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
                    zoom_b = gr.Slider(
                        minimum=50,
                        maximum=200,
                        value=100,
                        step=10,
                        label="Zoom (%)",
                        elem_id="zoom-b-slider",
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
                    sync_scrolling = gr.Checkbox(
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
            export_json_btn = gr.Button("Export JSON")
            export_pdf_btn = gr.Button("Export PDF Report")
        
        # Store comparison result and selected diff
        comparison_result = gr.State()
        selected_diff_index = gr.State(value=None)
        
        # Store gallery data for scrolling
        gallery_a_data_state = gr.State()
        gallery_b_data_state = gr.State()
        pdf_a_path_state = gr.State()
        pdf_b_path_state = gr.State()
        alignment_data_state = gr.State()
        
        # Store zoom levels
        zoom_a_state = gr.State(value=100)
        zoom_b_state = gr.State(value=100)
        
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
        
        def toggle_viewer(use_sync, result: ComparisonResult, pdf_a_path, pdf_b_path, alignment_data):
            """Toggle between Gallery and Synchronized viewer."""
            logger.info(f"Toggle viewer called: use_sync={use_sync}, result={result is not None}, pdf_a_path={pdf_a_path}, pdf_b_path={pdf_b_path}")
            
            gallery_row_update = gr.update(visible=not use_sync)
            sync_container_update = gr.update(visible=use_sync)
            
            # If toggling to sync viewer and we have comparison data, use premium PDF viewer
            if use_sync and pdf_a_path and pdf_b_path:
                logger.info(f"Setting sync viewer PDFs: A={pdf_a_path}, B={pdf_b_path}")
                
                # Use the premium PDF viewer from sync_pdf_viewer module
                from visualization.custom_components.sync_pdf_viewer import (
                    _generate_pdf_html,
                    _add_scroll_sync_script,
                )
                
                # Generate alignment data for scroll sync
                alignment_data_dict = alignment_data if alignment_data else {}
                sync_script = _add_scroll_sync_script(alignment_data_dict)
                
                # Generate premium HTML for both viewers
                sync_viewer_a_update = gr.update(value=_generate_pdf_html("a", pdf_a_path, sync_script))
                sync_viewer_b_update = gr.update(value=_generate_pdf_html("b", pdf_b_path, sync_script))
                
                logger.info(f"Premium PDF viewers generated for A and B")
            else:
                if use_sync:
                    logger.warning(f"Sync viewer enabled but missing PDF paths")
                sync_viewer_a_update = gr.update(value="<div>No PDF loaded</div>")
                sync_viewer_b_update = gr.update(value="<div>No PDF loaded</div>")
            
            return (
                gallery_row_update,
                sync_container_update,
            sync_viewer_a_update,
                sync_viewer_b_update,
            )
        
        def compare_documents(
            file_a, file_b, show_heat, scanned_mode, 
            force_ocr_mode,
            sensitivity,
            use_sync_viewer_flag,
            selected_ocr_engine,
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
                if selected_ocr_engine:
                    settings.ocr_engine = selected_ocr_engine
                    logger.info("OCR engine set to: %s", selected_ocr_engine)
                
                # Configure OCR based on scanned mode and force OCR
                use_ocr = scanned_mode or force_ocr_mode
                if scanned_mode:
                    # Scanned mode: force OCR and prioritize visual comparison
                    settings.use_ocr_for_all_documents = True
                    settings.ocr_enhancement_mode = "ocr_only"
                    status_msg = "Extracting scanned documents with OCR..."
                elif force_ocr_mode:
                    settings.use_ocr_for_all_documents = True
                    settings.ocr_enhancement_mode = "ocr_only"
                    status_msg = "Extracting documents with OCR enhancement..."
                else:
                    settings.use_ocr_for_all_documents = False
                    settings.ocr_enhancement_mode = "auto"
                    status_msg = "Extracting documents..."
                
                yield (
                    status_msg, None, None, None, 
                    gr.update(choices=[], value=None), None, None, None, None, 
                    None, None, None, None, None, None, ""
                )
                
                # Extract documents
                extraction_time = 0.0
                with track_time("extraction") as timing:
                    pages_a = extract_pdf(path_a, force_ocr=use_ocr)
                    pages_b = extract_pdf(path_b, force_ocr=use_ocr)
                    extraction_time = timing.duration
                
                total_pages = max(len(pages_a), len(pages_b))
                status_msg = f"Extracted {len(pages_a)} and {len(pages_b)} pages. Comparing..."
                yield (
                    status_msg, None, None, None, 
                    gr.update(choices=[], value=None), None, None, None, None, 
                    None, None, None, None, None, None, ""
                )
                
                # Compare text
                text_time = 0.0
                with track_time("text_comparison") as timing:
                    comparator = TextComparator()
                    text_diffs = comparator.compare(pages_a, pages_b)
                    text_time = timing.duration
                
                # Compare formatting
                formatting_time = 0.0
                with track_time("formatting_comparison") as timing:
                    formatting_diffs = compare_formatting(pages_a, pages_b)
                    formatting_time = timing.duration
                
                # Compare tables
                table_time = 0.0
                with track_time("table_comparison") as timing:
                    table_diffs = compare_tables(pages_a, pages_b)
                    table_time = timing.duration
                
                # Compare headers/footers
                header_footer_time = 0.0
                with track_time("header_footer_comparison") as timing:
                    header_footer_diffs = compare_headers_footers(pages_a, pages_b)
                    header_footer_time = timing.duration
                
                # Compare figure captions
                figure_time = 0.0
                with track_time("figure_comparison") as timing:
                    figure_diffs = compare_figure_captions(pages_a, pages_b)
                    figure_time = timing.duration
                
                # Combine and classify
                classification_time = 0.0
                with track_time("classification") as timing:
                    all_diffs = text_diffs + formatting_diffs + table_diffs + header_footer_diffs + figure_diffs
                    classified_diffs = classify_diffs(all_diffs)
                    classification_time = timing.duration
                
                # Create comparison result
                result = ComparisonResult(
                    doc1=str(path_a),
                    doc2=str(path_b),
                    pages=pages_a,  # Store pages from doc1
                    diffs=classified_diffs,
                    summary=get_diff_summary(classified_diffs),
                )
                
                status_msg = f"Found {len(classified_diffs)} differences. Rendering..."
                yield (
                    status_msg, None, None, None, 
                    gr.update(choices=[], value=None), result, None, None, None, 
                    None, None, None, None, None, None, ""
                )
                
                # Render pages with highlights (numpy arrays for Gallery)
                rendering_time = 0.0
                try:
                    with track_time("rendering") as timing:
                        rendered_a = render_pages(path_a, diffs=classified_diffs)
                        rendered_b = render_pages(path_b, diffs=classified_diffs)
                        rendering_time = timing.duration
                except Exception as exc:
                    logger.exception("Failed to render PDF pages: %s", exc)
                    # Fallback: render without diffs
                    try:
                        with track_time("rendering") as timing:
                            rendered_a = render_pages(path_a, diffs=None)
                            rendered_b = render_pages(path_b, diffs=None)
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
                
                # Prepare alignment data for sync viewer
                alignment_data = {}
                if use_sync_viewer_flag:
                    # Extract alignment information from comparison
                    alignment_data = {
                        "pages": len(pages_a),
                        "alignment_map": {},  # Can be enhanced with actual alignment data
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
                
                # Get OCR engine info if OCR was used
                ocr_engine_info = ""
                if use_ocr:
                    try:
                        from extraction.ocr_router import get_ocr_engine_name
                        # Try to get OCR engine from first page metadata if available
                        if pages_a and pages_a[0].blocks:
                            engine = pages_a[0].blocks[0].metadata.get("ocr_engine", "Unknown")
                            ocr_engine_info = f"**OCR Engine:** {engine}\n"
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
                    "total_pages": total_pages,
                    "total_changes": len(classified_diffs),
                    "low_confidence_count": low_confidence_count,
                    "changes_by_type": changes_by_type,
                }
                
                # Format performance info
                perf_info_text = f"""
### Performance Summary
- **Total Processing Time:** {total_time:.2f}s
- **Time per Page:** {time_per_page:.2f}s
- **Pages Processed:** {total_pages}

### Timing Breakdown
- Extraction: {extraction_time:.2f}s
- Text Comparison: {text_time:.2f}s
- Formatting Comparison: {formatting_time:.2f}s
- Table Comparison: {table_time:.2f}s
- Header/Footer Comparison: {header_footer_time:.2f}s
- Figure Comparison: {figure_time:.2f}s
- Classification: {classification_time:.2f}s
- Rendering: {rendering_time:.2f}s

### Changes Detected
- **Total:** {len(classified_diffs)} changes
  - Content: {changes_by_type['content']}
  - Formatting: {changes_by_type['formatting']}
  - Layout: {changes_by_type['layout']}
  - Visual: {changes_by_type['visual']}
- **Low Confidence:** {low_confidence_count} (may need review)

{ocr_engine_info}
"""
                
                status_msg = f"Comparison complete! Found {len(classified_diffs)} differences."
                
                # Generate iframe HTML for sync viewers using Gradio's file serving
                if use_sync_viewer_flag:
                    # Gradio serves files at /file={absolute_path}
                    iframe_a_html = f'''
                    <iframe
                        src="/file={str(path_a)}"
                        style="width: 100%; height: 800px; border: 1px solid #ccc;"
                        type="application/pdf"
                    >
                        <p>Your browser doesn't support PDFs. <a href="/file={str(path_a)}">Download PDF A</a></p>
                    </iframe>
                    '''
                    
                    iframe_b_html = f'''
                    <iframe
                        src="/file={str(path_b)}"
                        style="width: 100%; height: 800px; border: 1px solid #ccc;"
                        type="application/pdf"
                    >
                        <p>Your browser doesn't support PDFs. <a href="/file={str(path_b)}">Download PDF B</a></p>
                    </iframe>
                    '''
                    
                    sync_viewer_a_html = iframe_a_html
                    sync_viewer_b_html = iframe_b_html
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
        
        def get_severity(confidence: float) -> str:
            """Get severity indicator based on confidence score."""
            if confidence >= 0.7:
                return "🔴 High"
            elif confidence >= 0.4:
                return "🟡 Medium"
            else:
                return "🟢 Low"
        
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
            """Export comparison result as JSON."""
            if not result:
                return "No comparison result available."
            
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                    output_path = export_json(result, f.name)
                    return str(output_path)
            except Exception as exc:
                logger.exception("JSON export failed")
                return f"Export failed: {str(exc)}"
        
        def export_pdf_handler(result: ComparisonResult):
            """Export comparison result as PDF."""
            if not result:
                return "No comparison result available."
            
            # Validate that result is actually a ComparisonResult object
            if not isinstance(result, ComparisonResult):
                logger.error("Expected ComparisonResult, got %s", type(result))
                return f"Export failed: Invalid comparison result type: {type(result)}"
            
            # Ensure result has required attributes
            if not hasattr(result, 'doc1') or not hasattr(result, 'diffs'):
                logger.error("ComparisonResult missing required attributes")
                return "Export failed: Invalid comparison result structure."
            
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as f:
                    output_path = export_pdf(result, f.name)
                    return str(output_path)
            except Exception as exc:
                logger.exception("PDF export failed")
                return f"Export failed: {str(exc)}"
        
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
                    _add_scroll_sync_script,
                )
                sync_script = _add_scroll_sync_script(alignment_data or {})
                sync_a_update = gr.update(value=_generate_pdf_html("a", pdf_a_path, sync_script, page_num=page_num))
                sync_b_update = gr.update(value=_generate_pdf_html("b", pdf_b_path, sync_script, page_num=page_num))
            
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
                    _add_scroll_sync_script,
                )
                sync_script = _add_scroll_sync_script(alignment_data or {})
                sync_a_update = gr.update(value=_generate_pdf_html("a", pdf_a_path, sync_script, page_num=page_num))
                sync_b_update = gr.update(value=_generate_pdf_html("b", pdf_b_path, sync_script, page_num=page_num))
            
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
        use_sync_viewer.change(
            toggle_viewer,
            inputs=[use_sync_viewer, comparison_result, pdf_a_path_state, pdf_b_path_state, alignment_data_state],
            outputs=[gallery_row, sync_viewer_container, sync_viewer_a, sync_viewer_b],
        )
        
        # Show/hide OCR engine priority based on Force OCR checkbox
        def toggle_ocr_priority(force_ocr_enabled, scanned_mode):
            """Show OCR engine priority dropdown only when Force OCR or Scanned Mode is enabled."""
            show_priority = force_ocr_enabled or scanned_mode
            return gr.update(visible=show_priority)
        
        force_ocr.change(
            toggle_ocr_priority,
            inputs=[force_ocr, scanned_document_mode],
            outputs=[ocr_engine],
        )
        
        scanned_document_mode.change(
            toggle_ocr_priority,
            inputs=[force_ocr, scanned_document_mode],
            outputs=[ocr_engine],
        )
        
        compare_btn.click(
            compare_documents,
            inputs=[
                doc1, doc2, show_heatmap, scanned_document_mode, 
                force_ocr, sensitivity_threshold, use_sync_viewer, ocr_engine
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
        
        # Zoom controls (stored in state for persistence, applied via JavaScript)
        zoom_a.change(
            lambda z: z,
            inputs=[zoom_a],
            outputs=[zoom_a_state],
        )
        
        zoom_b.change(
            lambda z: z,
            inputs=[zoom_b],
            outputs=[zoom_b_state],
        )
        
        export_json_btn.click(
            export_json_handler,
            inputs=[comparison_result],
            outputs=[status],
        )
        
        export_pdf_btn.click(
            export_pdf_handler,
            inputs=[comparison_result],
            outputs=[status],
        )
    
    return demo
