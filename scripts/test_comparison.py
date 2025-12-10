"""Test comparison modules with downloaded models."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logging import configure_logging, logger

configure_logging()


def test_comparison_imports() -> bool:
    """Test that all comparison modules can be imported."""
    logger.info("Testing comparison module imports...")
    try:
        from comparison.alignment import align_pages, align_sections
        from comparison.diff_classifier import classify_diffs, get_diff_summary
        from comparison.figure_comparison import compare_figure_captions, extract_figure_captions
        from comparison.formatting_comparison import compare_formatting
        from comparison.models import Diff, PageData, TextBlock, Style
        from comparison.table_comparison import compare_tables
        from comparison.text_comparison import TextComparator
        from comparison.visual_diff import generate_heatmap, generate_heatmap_bytes
        
        logger.info("  ✓ alignment imported")
        logger.info("  ✓ diff_classifier imported")
        logger.info("  ✓ figure_comparison imported")
        logger.info("  ✓ formatting_comparison imported")
        logger.info("  ✓ models imported")
        logger.info("  ✓ table_comparison imported")
        logger.info("  ✓ text_comparison imported")
        logger.info("  ✓ visual_diff imported")
        
        logger.info("✓ Comparison module imports test PASSED")
        return True
    except Exception as exc:
        logger.error("✗ Comparison module imports test FAILED: %s", exc)
        import traceback
        logger.debug(traceback.format_exc())
        return False


def test_text_comparator_model() -> bool:
    """Test TextComparator with Sentence Transformer model."""
    logger.info("Testing TextComparator with Sentence Transformer...")
    try:
        from comparison.text_comparison import TextComparator
        from config.settings import settings
        
        logger.info("  Creating TextComparator with model: %s", settings.sentence_transformer_model)
        comparator = TextComparator()
        
        if comparator.model is None:
            logger.error("  ✗ Model is None")
            return False
        
        logger.info("  ✓ Model loaded successfully")
        
        # Test similarity computation
        text1 = "The quick brown fox jumps over the lazy dog"
        text2 = "A quick brown fox jumps over a lazy dog"
        similarity = comparator.similarity(text1, text2)
        logger.info("  ✓ Similarity computation works (similarity: %.3f)", similarity)
        
        # Test batch similarity
        text_pairs = [
            ("Hello world", "Hello world"),
            ("The cat sat", "A cat was sitting"),
            ("Different text", "Completely different"),
        ]
        batch_similarities = comparator.similarity_batch(text_pairs)
        logger.info("  ✓ Batch similarity works (results: %s)", [f"{s:.3f}" for s in batch_similarities])
        
        logger.info("✓ TextComparator model test PASSED")
        return True
    except Exception as exc:
        logger.error("✗ TextComparator model test FAILED: %s", exc)
        import traceback
        logger.debug(traceback.format_exc())
        return False


def test_alignment_functions() -> bool:
    """Test alignment functions."""
    logger.info("Testing alignment functions...")
    try:
        from comparison.alignment import align_pages, align_sections
        from comparison.models import PageData, TextBlock
        
        # Create test pages
        page1 = PageData(page_num=1, width=600, height=800)
        page1.blocks.append(TextBlock(text="Test block 1", bbox={"x": 10, "y": 10, "width": 100, "height": 20}))
        
        page2 = PageData(page_num=1, width=600, height=800)
        page2.blocks.append(TextBlock(text="Test block 1", bbox={"x": 10, "y": 10, "width": 100, "height": 20}))
        
        # Test page alignment
        alignment_map = align_pages([page1], [page2], use_similarity=False)
        logger.info("  ✓ align_pages works (result: %s)", alignment_map)
        
        # Test section alignment
        block_alignment = align_sections(page1, page2)
        logger.info("  ✓ align_sections works (result: %s)", block_alignment)
        
        logger.info("✓ Alignment functions test PASSED")
        return True
    except Exception as exc:
        logger.error("✗ Alignment functions test FAILED: %s", exc)
        import traceback
        logger.debug(traceback.format_exc())
        return False


def test_diff_classifier() -> bool:
    """Test diff classifier."""
    logger.info("Testing diff classifier...")
    try:
        from comparison.diff_classifier import classify_diffs, get_diff_summary
        from comparison.models import Diff
        
        # Create test diffs
        test_diffs = [
            Diff(
                page_num=1,
                diff_type="modified",
                change_type="content",
                old_text="Hello",
                new_text="World",
                bbox={"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.05},
                confidence=0.8,
            ),
            Diff(
                page_num=1,
                diff_type="added",
                change_type="content",
                old_text=None,
                new_text="New text",
                bbox={"x": 0.2, "y": 0.2, "width": 0.3, "height": 0.05},
                confidence=0.9,
            ),
        ]
        
        classified = classify_diffs(test_diffs)
        logger.info("  ✓ classify_diffs works (classified %d diffs)", len(classified))
        
        summary = get_diff_summary(classified)
        logger.info("  ✓ get_diff_summary works (summary: %s)", summary)
        
        logger.info("✓ Diff classifier test PASSED")
        return True
    except Exception as exc:
        logger.error("✗ Diff classifier test FAILED: %s", exc)
        import traceback
        logger.debug(traceback.format_exc())
        return False


def test_formatting_comparison() -> bool:
    """Test formatting comparison."""
    logger.info("Testing formatting comparison...")
    try:
        from comparison.formatting_comparison import compare_formatting
        from comparison.models import PageData, TextBlock, Style
        
        # Create test pages with different formatting
        page1 = PageData(page_num=1, width=600, height=800)
        page1.blocks.append(TextBlock(
            text="Same text",
            bbox={"x": 10, "y": 10, "width": 100, "height": 20},
            style=Style(font="Arial", size=12.0, bold=False)
        ))
        
        page2 = PageData(page_num=1, width=600, height=800)
        page2.blocks.append(TextBlock(
            text="Same text",
            bbox={"x": 10, "y": 10, "width": 100, "height": 20},
            style=Style(font="Times", size=14.0, bold=True)
        ))
        
        diffs = compare_formatting([page1], [page2])
        logger.info("  ✓ compare_formatting works (found %d diffs)", len(diffs))
        
        logger.info("✓ Formatting comparison test PASSED")
        return True
    except Exception as exc:
        logger.error("✗ Formatting comparison test FAILED: %s", exc)
        import traceback
        logger.debug(traceback.format_exc())
        return False


def test_table_comparison() -> bool:
    """Test table comparison."""
    logger.info("Testing table comparison...")
    try:
        from comparison.table_comparison import compare_tables
        from comparison.models import PageData
        
        # Create test pages with table metadata
        page1 = PageData(page_num=1, width=600, height=800)
        page1.metadata = {
            "tables": [{
                "bbox": [10, 10, 200, 100],
                "confidence": 0.8
            }]
        }
        
        page2 = PageData(page_num=1, width=600, height=800)
        page2.metadata = {
            "tables": [{
                "bbox": [10, 10, 200, 100],
                "confidence": 0.8
            }]
        }
        
        diffs = compare_tables([page1], [page2])
        logger.info("  ✓ compare_tables works (found %d diffs)", len(diffs))
        
        logger.info("✓ Table comparison test PASSED")
        return True
    except Exception as exc:
        logger.error("✗ Table comparison test FAILED: %s", exc)
        import traceback
        logger.debug(traceback.format_exc())
        return False


def test_figure_comparison() -> bool:
    """Test figure comparison."""
    logger.info("Testing figure comparison...")
    try:
        from comparison.figure_comparison import extract_figure_captions, compare_figure_captions
        from comparison.models import PageData
        
        # Create test pages with figure metadata
        page1 = PageData(page_num=1, width=600, height=800)
        page1.metadata = {
            "figures": [{
                "bbox": [10, 10, 200, 150],
                "xref": 1,
                "width": 190,
                "height": 140,
                "confidence": 1.0
            }]
        }
        # PageData already has blocks as default_factory, so it's fine
        
        captions = extract_figure_captions(page1)
        logger.info("  ✓ extract_figure_captions works (found %d captions)", len(captions))
        
        diffs = compare_figure_captions([page1], [page1])
        logger.info("  ✓ compare_figure_captions works (found %d diffs)", len(diffs))
        
        logger.info("✓ Figure comparison test PASSED")
        return True
    except Exception as exc:
        logger.error("✗ Figure comparison test FAILED: %s", exc)
        import traceback
        logger.debug(traceback.format_exc())
        return False


def test_visual_diff() -> bool:
    """Test visual diff functions."""
    logger.info("Testing visual diff functions...")
    try:
        from comparison.visual_diff import generate_heatmap, generate_heatmap_bytes
        
        # Just test that functions can be imported and have correct signatures
        import inspect
        
        sig1 = inspect.signature(generate_heatmap)
        logger.info("  ✓ generate_heatmap signature: %s", sig1)
        
        sig2 = inspect.signature(generate_heatmap_bytes)
        logger.info("  ✓ generate_heatmap_bytes signature: %s", sig2)
        
        logger.info("✓ Visual diff test PASSED (functions available)")
        return True
    except Exception as exc:
        logger.error("✗ Visual diff test FAILED: %s", exc)
        import traceback
        logger.debug(traceback.format_exc())
        return False


def test_models() -> bool:
    """Test comparison models."""
    logger.info("Testing comparison models...")
    try:
        from comparison.models import Diff, PageData, TextBlock, Style, ComparisonResult
        
        # Test Style
        style = Style(font="Arial", size=12.0, bold=True, italic=False, color=(255, 0, 0))
        logger.info("  ✓ Style model works")
        
        # Test TextBlock
        block = TextBlock(
            text="Test",
            bbox={"x": 10, "y": 10, "width": 100, "height": 20},
            style=style
        )
        normalized = block.normalize_bbox(600, 800)
        logger.info("  ✓ TextBlock model works (normalized bbox: %s)", normalized)
        
        # Test PageData
        page = PageData(page_num=1, width=600, height=800, blocks=[block])
        logger.info("  ✓ PageData model works")
        
        # Test Diff
        diff = Diff(
            page_num=1,
            diff_type="modified",
            change_type="content",
            old_text="Old",
            new_text="New",
            bbox={"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.05},
            confidence=0.8
        )
        logger.info("  ✓ Diff model works")
        
        # Test ComparisonResult
        result = ComparisonResult(doc1="doc1.pdf", doc2="doc2.pdf", pages=[page], diffs=[diff])
        logger.info("  ✓ ComparisonResult model works")
        
        logger.info("✓ Models test PASSED")
        return True
    except Exception as exc:
        logger.error("✗ Models test FAILED: %s", exc)
        import traceback
        logger.debug(traceback.format_exc())
        return False


def test_integration_with_models() -> bool:
    """Test that comparison modules work together with models."""
    logger.info("Testing integration with models...")
    try:
        from comparison.text_comparison import TextComparator
        from comparison.models import PageData, TextBlock
        from config.settings import settings
        
        # Create test pages
        page1 = PageData(page_num=1, width=600, height=800)
        page1.blocks.append(TextBlock(
            text="The cat sat on the mat",
            bbox={"x": 10, "y": 10, "width": 200, "height": 20}
        ))
        
        page2 = PageData(page_num=1, width=600, height=800)
        page2.blocks.append(TextBlock(
            text="A cat was sitting on the mat",
            bbox={"x": 10, "y": 10, "width": 200, "height": 20}
        ))
        
        # Test text comparison with model
        comparator = TextComparator()
        diffs = comparator.compare([page1], [page2])
        logger.info("  ✓ Text comparison with model works (found %d diffs)", len(diffs))
        
        # Verify model path is correct
        if settings.sentence_transformer_model.startswith("models/"):
            logger.info("  ✓ Using local model: %s", settings.sentence_transformer_model)
        else:
            logger.info("  ℹ Using HuggingFace model: %s", settings.sentence_transformer_model)
        
        logger.info("✓ Integration test PASSED")
        return True
    except Exception as exc:
        logger.error("✗ Integration test FAILED: %s", exc)
        import traceback
        logger.debug(traceback.format_exc())
        return False


def main() -> None:
    """Run all comparison module tests."""
    logger.info("=" * 60)
    logger.info("Comparison Modules Test Suite")
    logger.info("=" * 60)
    logger.info("")
    
    results = {}
    
    # Test 1: Module imports
    logger.info("Test 1: Module Imports")
    logger.info("-" * 60)
    results['imports'] = test_comparison_imports()
    logger.info("")
    
    # Test 2: Models
    logger.info("Test 2: Data Models")
    logger.info("-" * 60)
    results['models'] = test_models()
    logger.info("")
    
    # Test 3: Text Comparator with Sentence Transformer
    logger.info("Test 3: Text Comparator (Sentence Transformer)")
    logger.info("-" * 60)
    results['text_comparator'] = test_text_comparator_model()
    logger.info("")
    
    # Test 4: Alignment
    logger.info("Test 4: Alignment Functions")
    logger.info("-" * 60)
    results['alignment'] = test_alignment_functions()
    logger.info("")
    
    # Test 5: Diff Classifier
    logger.info("Test 5: Diff Classifier")
    logger.info("-" * 60)
    results['diff_classifier'] = test_diff_classifier()
    logger.info("")
    
    # Test 6: Formatting Comparison
    logger.info("Test 6: Formatting Comparison")
    logger.info("-" * 60)
    results['formatting'] = test_formatting_comparison()
    logger.info("")
    
    # Test 7: Table Comparison
    logger.info("Test 7: Table Comparison")
    logger.info("-" * 60)
    results['table'] = test_table_comparison()
    logger.info("")
    
    # Test 8: Figure Comparison
    logger.info("Test 8: Figure Comparison")
    logger.info("-" * 60)
    results['figure'] = test_figure_comparison()
    logger.info("")
    
    # Test 9: Visual Diff
    logger.info("Test 9: Visual Diff")
    logger.info("-" * 60)
    results['visual'] = test_visual_diff()
    logger.info("")
    
    # Test 10: Integration
    logger.info("Test 10: Integration with Models")
    logger.info("-" * 60)
    results['integration'] = test_integration_with_models()
    logger.info("")
    
    # Summary
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info("%s: %s", test_name.replace('_', ' ').title(), status)
    
    logger.info("")
    if all_passed:
        logger.info("✓ All comparison module tests PASSED!")
        logger.info("")
        logger.info("Models are properly integrated:")
        logger.info("  - Sentence Transformer (MiniLM): Working in TextComparator")
        logger.info("  - All comparison modules: Functional")
        sys.exit(0)
    else:
        logger.error("✗ Some tests failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

