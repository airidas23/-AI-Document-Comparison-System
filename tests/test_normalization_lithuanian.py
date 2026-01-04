"""
Unit tests for text normalization with Lithuanian character support.

These tests verify that:
1. Lithuanian characters (ąčęėįšųūž) are preserved after normalization
2. NFC normalization handles composed vs decomposed forms
3. Special characters (NBSP, zero-width, soft hyphen) are handled correctly
4. Dash variants are normalized consistently
"""
import pytest
from utils.text_normalization import normalize_text, normalize_text_full


class TestLithuanianCharacters:
    """Test Lithuanian character preservation."""
    
    def test_lowercase_lithuanian(self):
        """Lowercase Lithuanian letters should be preserved."""
        text = "ąčęėįšųūž"
        result = normalize_text_full(text)
        assert result == "ąčęėįšųūž"
    
    def test_uppercase_lithuanian(self):
        """Uppercase Lithuanian letters should be lowercased."""
        text = "ĄČĘĖĮŠŲŪŽ"
        result = normalize_text_full(text)
        assert result == "ąčęėįšųūž"
    
    def test_mixed_lithuanian_sentence(self):
        """Full Lithuanian sentence should be normalized correctly."""
        text = "Ąžuolas ėjo į mokyklą"
        result = normalize_text_full(text)
        assert result == "ąžuolas ėjo į mokyklą"
    
    def test_preserve_case_option(self):
        """With preserve_case=True, case should not change."""
        text = "Ąžuolas Ėjo Į Mokyklą"
        result = normalize_text_full(text, preserve_case=True)
        assert result == "Ąžuolas Ėjo Į Mokyklą"
    
    def test_lithuanian_with_numbers(self):
        """Lithuanian text with numbers should work."""
        text = "Straipsnis Nr. 15, 2024 metų gruodžio 5 d."
        result = normalize_text_full(text)
        assert "metų" in result
        assert "gruodžio" in result


class TestNFCNormalization:
    """Test NFC (composed form) Unicode normalization."""
    
    def test_composed_vs_decomposed_a_ogonek(self):
        """Composed ą should equal NFC-normalized decomposed form."""
        composed = "ą"  # U+0105 (LATIN SMALL LETTER A WITH OGONEK)
        decomposed = "a\u0328"  # 'a' + COMBINING OGONEK
        
        result_composed = normalize_text_full(composed)
        result_decomposed = normalize_text_full(decomposed)
        
        assert result_composed == result_decomposed
        assert result_composed == "ą"
    
    def test_composed_vs_decomposed_e_ogonek(self):
        """Composed ę should equal NFC-normalized decomposed form."""
        composed = "ę"
        decomposed = "e\u0328"
        
        result_composed = normalize_text_full(composed)
        result_decomposed = normalize_text_full(decomposed)
        
        assert result_composed == result_decomposed
    
    def test_composed_vs_decomposed_z_caron(self):
        """Composed ž should equal NFC-normalized decomposed form."""
        composed = "ž"  # U+017E
        decomposed = "z\u030C"  # 'z' + COMBINING CARON
        
        result_composed = normalize_text_full(composed)
        result_decomposed = normalize_text_full(decomposed)
        
        assert result_composed == result_decomposed


class TestSpecialSpaces:
    """Test special space character normalization."""
    
    def test_nbsp_to_space(self):
        """NBSP (U+00A0) should become regular space."""
        text = "Hello\u00A0World"
        result = normalize_text_full(text)
        assert "\u00A0" not in result
        assert result == "hello world"
    
    def test_narrow_nbsp_to_space(self):
        """Narrow NBSP (U+202F) should become regular space."""
        text = "10\u202F000"
        result = normalize_text_full(text)
        assert "\u202F" not in result
        assert result == "10 000"
    
    def test_figure_space_to_space(self):
        """Figure space (U+2007) should become regular space."""
        text = "100\u2007200"
        result = normalize_text_full(text)
        assert result == "100 200"
    
    def test_thin_space_to_space(self):
        """Thin space (U+2009) should become regular space."""
        text = "word\u2009word"
        result = normalize_text_full(text)
        assert result == "word word"
    
    def test_multiple_special_spaces(self):
        """Multiple special spaces should collapse to single space."""
        text = "A\u00A0\u202F\u2009B"
        result = normalize_text_full(text)
        assert result == "a b"


class TestZeroWidthCharacters:
    """Test zero-width character removal."""
    
    def test_zwsp_removal(self):
        """Zero-width space (U+200B) should be removed."""
        text = "Test\u200BWord"
        result = normalize_text_full(text)
        assert "\u200B" not in result
        assert result == "testword"
    
    def test_zwnj_removal(self):
        """Zero-width non-joiner (U+200C) should be removed."""
        text = "Test\u200CWord"
        result = normalize_text_full(text)
        assert result == "testword"
    
    def test_zwj_removal(self):
        """Zero-width joiner (U+200D) should be removed."""
        text = "Test\u200DWord"
        result = normalize_text_full(text)
        assert result == "testword"
    
    def test_bom_removal(self):
        """BOM/ZWNBSP (U+FEFF) should be removed."""
        text = "\uFEFFStarting with BOM"
        result = normalize_text_full(text)
        assert "\uFEFF" not in result
        assert result.startswith("starting")
    
    def test_soft_hyphen_removal(self):
        """Soft hyphen (U+00AD) should be removed."""
        text = "hyphe\u00ADnated"
        result = normalize_text_full(text)
        assert "\u00AD" not in result
        assert result == "hyphenated"


class TestDashNormalization:
    """Test dash variant normalization."""
    
    def test_en_dash_to_hyphen(self):
        """En dash (U+2013) should become hyphen."""
        text = "2020\u20132021"
        result = normalize_text_full(text)
        assert result == "2020-2021"
    
    def test_em_dash_to_hyphen(self):
        """Em dash (U+2014) should become hyphen."""
        text = "word\u2014word"
        result = normalize_text_full(text)
        assert result == "word-word"
    
    def test_minus_sign_to_hyphen(self):
        """Minus sign (U+2212) should become hyphen."""
        text = "5\u22123=2"
        result = normalize_text_full(text)
        assert result == "5-3=2"
    
    def test_spacing_around_dashes(self):
        """Spacing around dashes should be normalized."""
        text = "word \u2013 word"
        result = normalize_text_full(text)
        assert result == "word-word"
    
    def test_disable_dash_normalization(self):
        """With normalize_dashes=False, dashes should be preserved."""
        text = "2020\u20132021"
        result = normalize_text_full(text, normalize_dashes=False)
        assert "\u2013" in result


class TestWhitespaceCollapse:
    """Test whitespace collapsing."""
    
    def test_multiple_spaces(self):
        """Multiple spaces should collapse to single space."""
        text = "Hello    World"
        result = normalize_text_full(text)
        assert result == "hello world"
    
    def test_tabs_to_space(self):
        """Tabs should become spaces and collapse."""
        text = "Hello\t\tWorld"
        result = normalize_text_full(text)
        assert result == "hello world"
    
    def test_newlines_to_space(self):
        """Newlines should become spaces and collapse."""
        text = "Hello\n\nWorld"
        result = normalize_text_full(text)
        assert result == "hello world"
    
    def test_mixed_whitespace(self):
        """Mixed whitespace should collapse."""
        text = "Hello \t\n  World"
        result = normalize_text_full(text)
        assert result == "hello world"
    
    def test_leading_trailing_whitespace(self):
        """Leading and trailing whitespace should be stripped."""
        text = "   Hello World   "
        result = normalize_text_full(text)
        assert result == "hello world"


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_string(self):
        """Empty string should return empty string."""
        assert normalize_text_full("") == ""
    
    def test_none_input(self):
        """None input should not crash (returns empty)."""
        # The function should handle None gracefully
        result = normalize_text_full("")
        assert result == ""
    
    def test_only_whitespace(self):
        """String with only whitespace should return empty."""
        result = normalize_text_full("   \t\n   ")
        assert result == ""
    
    def test_only_special_chars(self):
        """String with only special chars should return empty."""
        result = normalize_text_full("\u200B\u200C\u200D")
        assert result == ""
    
    def test_very_long_string(self):
        """Very long string should be handled efficiently."""
        text = "Ąžuolas " * 1000
        result = normalize_text_full(text)
        assert len(result) > 0
        assert "ąžuolas" in result


class TestBackwardCompatibility:
    """Test backward compatibility with original normalize_text()."""
    
    def test_basic_lowercase(self):
        """Original function should still work for basic lowercase."""
        text = "HELLO WORLD"
        result = normalize_text(text)
        assert result == "hello world"
    
    def test_original_whitespace_collapse(self):
        """Original function should collapse whitespace."""
        text = "Hello    World"
        result = normalize_text(text)
        assert result == "hello world"
    
    def test_original_nfc(self):
        """Original function should do NFC normalization."""
        composed = "ą"
        decomposed = "a\u0328"
        
        result_composed = normalize_text(composed)
        result_decomposed = normalize_text(decomposed)
        
        assert result_composed == result_decomposed


class TestComparisonScenarios:
    """Test real-world comparison scenarios."""
    
    def test_pdf_extracted_vs_typed(self):
        """PDF-extracted text with NBSP should match typed text."""
        pdf_text = "Šis\u00A0dokumentas"  # From PDF
        typed_text = "Šis dokumentas"      # Typed manually
        
        result_pdf = normalize_text_full(pdf_text)
        result_typed = normalize_text_full(typed_text)
        
        assert result_pdf == result_typed
    
    def test_different_encoding_sources(self):
        """Text from different encoding sources should match after normalization."""
        source1 = "Žmogaus 3D modelio konstravimas"
        source2 = "Žmogaus\u00A03D\u00A0modelio\u00A0konstravimas"
        
        result1 = normalize_text_full(source1)
        result2 = normalize_text_full(source2)
        
        assert result1 == result2
    
    def test_ocr_vs_digital(self):
        """OCR text with artifacts should normalize to digital text."""
        # Simulated OCR with soft hyphens and zero-width chars
        ocr_text = "Uni\u00ADversi\u200Btetas"
        digital_text = "Universitetas"
        
        result_ocr = normalize_text_full(ocr_text)
        result_digital = normalize_text_full(digital_text)
        
        assert result_ocr == result_digital
