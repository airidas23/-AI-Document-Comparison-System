#!/usr/bin/env python3
"""Test script to verify DocLayout-YOLO model loading and detection."""
import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from extraction.layout_analyzer import load_yolo_model, analyze_layout
from utils.logging import logger

def test_doclayout_yolo():
    """Test if DocLayout-YOLO loads and detects document-specific elements."""
    print("=" * 70)
    print("DocLayout-YOLO Model Test")
    print("=" * 70)
    
    # Test 1: Load the model
    print("\nTest 1: Loading YOLO model...")
    model = load_yolo_model()
    
    assert model is not None, "DocLayout-YOLO model failed to load"
    
    print(f"✅ Model loaded: {type(model)}")
    
    # Check model classes if available
    if hasattr(model, 'names'):
        print(f"\n📋 Model can detect {len(model.names)} classes:")
        for idx, name in model.names.items():
            print(f"   {idx}: {name}")
    
    # Test 2: Analyze a PDF
    print("\n" + "=" * 70)
    print("Test 2: Analyzing PDF with DocLayout-YOLO...")
    print("=" * 70)
    
    test_pdf = project_root / "AI Document Comparison System Prototype Plan.pdf"
    
    if not test_pdf.exists():
        print(f"⚠️  Test PDF not found: {test_pdf}")
        pytest.skip("Test PDF not available")
    
    try:
        pages = analyze_layout(test_pdf, use_layoutparser=True)
        print(f"\n✅ Successfully analyzed {len(pages)} pages")
        
        # Analyze detected elements
        if pages:
            method = pages[0].metadata.get("layout_method", "unknown")
            print(f"📊 Layout method: {method}")
            
            # Count detected elements across all pages
            total_tables = 0
            total_figures = 0
            total_text = 0
            
            for page in pages:
                tables = page.metadata.get("tables", [])
                figures = page.metadata.get("figures", [])
                text_blocks = page.metadata.get("text_blocks", [])
                
                total_tables += len(tables)
                total_figures += len(figures)
                total_text += len(text_blocks)
                
                # Show sample detections from first page
                if page.page_num == 1 and (tables or figures or text_blocks):
                    print(f"\n📄 Page {page.page_num} detections:")
                    if tables:
                        print(f"   Tables: {len(tables)}")
                        for i, table in enumerate(tables[:3]):  # Show first 3
                            print(f"      {i+1}. {table.get('label', 'unknown')} (confidence: {table.get('confidence', 0):.2f})")
                    if figures:
                        print(f"   Figures: {len(figures)}")
                        for i, fig in enumerate(figures[:3]):
                            print(f"      {i+1}. {fig.get('label', 'unknown')} (confidence: {fig.get('confidence', 0):.2f})")
                    if text_blocks:
                        print(f"   Text blocks: {len(text_blocks)}")
                        for i, text in enumerate(text_blocks[:3]):
                            print(f"      {i+1}. {text.get('label', 'unknown')} (confidence: {text.get('confidence', 0):.2f})")
            
            print(f"\n📊 Total across all pages:")
            print(f"   Tables: {total_tables}")
            print(f"   Figures: {total_figures}")
            print(f"   Text blocks: {total_text}")
            
            if method == "yolo" and (total_tables > 0 or total_figures > 0 or total_text > 0):
                print("\n✅ DocLayout-YOLO is working correctly!")
                assert True
            else:
                print(f"\n⚠️  Expected YOLO detections, but got method='{method}' with limited detections")
                assert method == "yolo", f"Expected layout_method='yolo', got '{method}'"
        
        assert True
    except Exception as e:
        print(f"❌ Error analyzing PDF: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Error analyzing PDF: {e}")

if __name__ == "__main__":
    success = test_doclayout_yolo()
    print("\n" + "=" * 70)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 70)
    sys.exit(0 if success else 1)
