"""OCR-aware normalization for academic documents.

Phase 2: Reduces phantom diffs by handling common OCR artifacts:
- Hyphenation (word-\nbreak → wordbreak)
- Ligatures (ﬁ, ﬂ → fi, fl)
- Diacritics near-miss (ė↔e, ū↔u as low severity)
- Citation noise ([12], (12), ¹)
- Whitespace/line-break soft equivalence

This module provides TWO normalization levels:
1. `normalize_ocr_strict()` - For exact matching (checksums, dedup)
2. `normalize_ocr_compare()` - For similarity comparison (preserves semantic content)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# =============================================================================
# Ligature Mappings
# =============================================================================

LIGATURE_MAP: Dict[str, str] = {
    "\ufb00": "ff",    # ﬀ
    "\ufb01": "fi",    # ﬁ
    "\ufb02": "fl",    # ﬂ
    "\ufb03": "ffi",   # ﬃ
    "\ufb04": "ffl",   # ﬄ
    "\ufb05": "st",    # ﬅ (long s + t)
    "\ufb06": "st",    # ﬆ
    "\u0152": "OE",    # Œ
    "\u0153": "oe",    # œ
    "\u00c6": "AE",    # Æ
    "\u00e6": "ae",    # æ
    "\u0132": "IJ",    # Ĳ
    "\u0133": "ij",    # ĳ
}


# =============================================================================
# Diacritics Near-Miss Mappings (Lithuanian + Common European)
# =============================================================================

# Maps accented characters to their base form for "soft" comparison
# This is used to detect near-miss OCR errors (ė vs e) as low severity
DIACRITICS_MAP: Dict[str, str] = {
    # Lithuanian
    "ą": "a", "Ą": "A",
    "č": "c", "Č": "C",
    "ę": "e", "Ę": "E",
    "ė": "e", "Ė": "E",
    "į": "i", "Į": "I",
    "š": "s", "Š": "S",
    "ų": "u", "Ų": "U",
    "ū": "u", "Ū": "U",
    "ž": "z", "Ž": "Z",
    # Polish
    "ó": "o", "Ó": "O",
    "ł": "l", "Ł": "L",
    "ń": "n", "Ń": "N",
    "ś": "s", "Ś": "S",
    "ź": "z", "Ź": "Z",
    "ż": "z", "Ż": "Z",
    # German
    "ä": "a", "Ä": "A",
    "ö": "o", "Ö": "O",
    "ü": "u", "Ü": "U",
    "ß": "ss",
    # French
    "é": "e", "É": "E",
    "è": "e", "È": "E",
    "ê": "e", "Ê": "E",
    "ë": "e", "Ë": "E",
    "à": "a", "À": "A",
    "â": "a", "Â": "A",
    "î": "i", "Î": "I",
    "ï": "i", "Ï": "I",
    "ô": "o", "Ô": "O",
    "ù": "u", "Ù": "U",
    "û": "u", "Û": "U",
    "ç": "c", "Ç": "C",
    # Spanish
    "ñ": "n", "Ñ": "N",
    "í": "i", "Í": "I",
    "ú": "u", "Ú": "U",
    # Nordic
    "å": "a", "Å": "A",
    "ø": "o", "Ø": "O",
    "æ": "ae", "Æ": "AE",
}


# =============================================================================
# Citation Patterns (Academic Documents)
# =============================================================================

# Regex patterns for citation markers that OCR often mangles
CITATION_PATTERNS = [
    r"\[\d+(?:[-–,]\d+)*\]",           # [12], [1-3], [1,2,3]
    r"\(\d+(?:[-–,]\d+)*\)",           # (12), (1-3), (1,2,3)
    r"(?<!\d)[¹²³⁴⁵⁶⁷⁸⁹⁰]+(?!\d)",   # Superscript numbers
    r"\[\w+(?:\s+et\s+al\.?)?,?\s*\d{4}\]",  # [Author, 2020] or [Author et al., 2020]
    r"\(\w+(?:\s+et\s+al\.?)?,?\s*\d{4}\)",  # (Author, 2020)
]

# Compiled citation regex for performance
_CITATION_REGEX = re.compile("|".join(CITATION_PATTERNS), re.UNICODE)


# =============================================================================
# Dash/Hyphen Variants
# =============================================================================

DASH_CHARS = {
    "\u2010",  # Hyphen
    "\u2011",  # Non-Breaking Hyphen
    "\u2012",  # Figure Dash
    "\u2013",  # En Dash
    "\u2014",  # Em Dash
    "\u2015",  # Horizontal Bar
    "\u2212",  # Minus Sign
    "\u00AD",  # Soft Hyphen
}


# =============================================================================
# Quote Variants
# =============================================================================

QUOTE_PAIRS = {
    # Opening quotes → standard
    "\u201C": '"',  # "
    "\u201D": '"',  # "
    "\u201E": '"',  # „
    "\u201F": '"',  # ‟
    "\u2018": "'",  # '
    "\u2019": "'",  # '
    "\u201A": "'",  # ‚
    "\u201B": "'",  # ‛
    "\u00AB": '"',  # «
    "\u00BB": '"',  # »
    "\u2039": "'",  # ‹
    "\u203A": "'",  # ›
}


# =============================================================================
# OCR Normalization Configuration
# =============================================================================

@dataclass
class OCRNormalizationConfig:
    """Configuration for OCR normalization behavior."""
    
    # Ligature handling
    expand_ligatures: bool = True
    
    # Diacritics handling
    strip_diacritics: bool = False  # Only for near-miss detection, not default
    diacritics_as_low_severity: bool = True
    
    # Hyphenation handling
    merge_hyphenated_words: bool = True
    
    # Citation handling
    normalize_citations: bool = True
    strip_citations: bool = False  # Only for strict matching
    
    # Whitespace handling
    collapse_whitespace: bool = True
    normalize_line_breaks: bool = True
    soft_line_break_equivalence: bool = True
    
    # Quote/dash normalization
    normalize_quotes: bool = True
    normalize_dashes: bool = True
    
    # Case handling
    lowercase: bool = False  # Only for certain comparisons
    
    # Academic-specific
    normalize_page_numbers: bool = True
    strip_page_numbers: bool = False
    
    @classmethod
    def strict(cls) -> "OCRNormalizationConfig":
        """Strict normalization for checksums and deduplication."""
        return cls(
            expand_ligatures=True,
            strip_diacritics=False,
            merge_hyphenated_words=True,
            normalize_citations=True,
            strip_citations=True,  # Remove for checksum
            collapse_whitespace=True,
            normalize_line_breaks=True,
            soft_line_break_equivalence=True,
            normalize_quotes=True,
            normalize_dashes=True,
            lowercase=True,
            normalize_page_numbers=True,
            strip_page_numbers=True,
        )
    
    @classmethod
    def compare(cls) -> "OCRNormalizationConfig":
        """Comparison normalization (preserves semantic content)."""
        return cls(
            expand_ligatures=True,
            strip_diacritics=False,
            diacritics_as_low_severity=True,
            merge_hyphenated_words=True,
            normalize_citations=True,
            strip_citations=False,  # Keep citations for comparison
            collapse_whitespace=True,
            normalize_line_breaks=True,
            soft_line_break_equivalence=True,
            normalize_quotes=True,
            normalize_dashes=True,
            lowercase=False,  # Preserve case
            normalize_page_numbers=True,
            strip_page_numbers=False,
        )
    
    @classmethod
    def minimal(cls) -> "OCRNormalizationConfig":
        """Minimal normalization (only essential cleanup)."""
        return cls(
            expand_ligatures=True,
            strip_diacritics=False,
            merge_hyphenated_words=False,
            normalize_citations=False,
            collapse_whitespace=True,
            normalize_line_breaks=False,
            soft_line_break_equivalence=False,
            normalize_quotes=False,
            normalize_dashes=False,
            lowercase=False,
        )


# =============================================================================
# Core Normalization Functions
# =============================================================================

def expand_ligatures(text: str) -> str:
    """Expand typographic ligatures to their component letters.
    
    Args:
        text: Input text potentially containing ligatures
        
    Returns:
        Text with ligatures expanded (ﬁ → fi, ﬂ → fl, etc.)
    """
    for ligature, expansion in LIGATURE_MAP.items():
        text = text.replace(ligature, expansion)
    return text


def strip_diacritics(text: str) -> str:
    """Remove diacritical marks from text.
    
    Uses both explicit mapping and Unicode decomposition for comprehensive coverage.
    
    Args:
        text: Input text with potential diacritics
        
    Returns:
        Text with diacritics removed (ąčęėįšųūž → aceeisuuz)
    """
    # First apply explicit mappings for known characters
    for accented, base in DIACRITICS_MAP.items():
        text = text.replace(accented, base)
    
    # Then use Unicode decomposition for any remaining
    # NFD decomposes characters, then we filter combining marks
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if not unicodedata.combining(c))


def merge_hyphenated_words(text: str) -> str:
    """Merge words split across lines by hyphenation.
    
    Handles patterns like:
    - "infor-\\nmacija" → "informacija"
    - "infor- macija" → "informacija"
    - "infor-\\n  macija" → "informacija"
    
    Args:
        text: Input text with potential hyphenation
        
    Returns:
        Text with hyphenated words merged
    """
    # Pattern: hyphen (including variants) followed by optional whitespace and newline
    # then optional whitespace, then lowercase continuation
    pattern = r"([a-zA-ZąčęėįšųūžĄČĘĖĮŠŲŪŽ])[-\u2010\u2011\u00AD]\s*[\n\r]+\s*([a-zA-ZąčęėįšųūžĄČĘĖĮŠŲŪŽ])"
    
    # Keep merging until no more matches (handles multiple hyphenations)
    prev_text = None
    while prev_text != text:
        prev_text = text
        text = re.sub(pattern, r"\1\2", text)
    
    return text


def normalize_citations(text: str, strip: bool = False) -> str:
    """Normalize or strip citation markers.
    
    Args:
        text: Input text with citations
        strip: If True, remove citations entirely. If False, normalize to [#].
        
    Returns:
        Text with normalized or stripped citations
    """
    if strip:
        return _CITATION_REGEX.sub("", text)
    
    # Normalize various citation formats to [#]
    # This helps match "[12]" with "(12)" or "¹²"
    def normalize_match(m: re.Match) -> str:
        match_text = m.group(0)
        # Extract numbers
        numbers = re.findall(r"\d+", match_text)
        if numbers:
            return "[" + ",".join(numbers) + "]"
        return match_text
    
    return _CITATION_REGEX.sub(normalize_match, text)


def normalize_quotes(text: str) -> str:
    """Normalize quote characters to ASCII equivalents.
    
    Args:
        text: Input text with fancy quotes
        
    Returns:
        Text with normalized quotes (" and ')
    """
    for fancy, simple in QUOTE_PAIRS.items():
        text = text.replace(fancy, simple)
    return text


def normalize_dashes(text: str) -> str:
    """Normalize dash variants to ASCII hyphen.
    
    Args:
        text: Input text with various dash characters
        
    Returns:
        Text with all dashes as ASCII hyphen (-)
    """
    for dash in DASH_CHARS:
        text = text.replace(dash, "-")
    return text


def normalize_whitespace(text: str, soft_line_breaks: bool = True) -> str:
    """Normalize whitespace for OCR comparison.
    
    Args:
        text: Input text
        soft_line_breaks: If True, treat line breaks as spaces
        
    Returns:
        Text with normalized whitespace
    """
    if soft_line_breaks:
        # Replace line breaks with spaces first
        text = re.sub(r"[\n\r]+", " ", text)
    
    # Collapse multiple whitespace to single space
    text = re.sub(r"[ \t]+", " ", text)
    
    # Strip leading/trailing
    text = text.strip()
    
    return text


def strip_page_numbers(text: str) -> str:
    """Remove page number patterns from text.
    
    Handles patterns like:
    - "Page 12", "page 12", "p. 12", "- 12 -", "12 of 100"
    
    Args:
        text: Input text potentially containing page numbers
        
    Returns:
        Text with page numbers removed
    """
    patterns = [
        r"\b[Pp]age\s+\d+\b",
        r"\b[Pp]\.\s*\d+\b",
        r"\b\d+\s+of\s+\d+\b",
        r"^\s*[-–—]\s*\d+\s*[-–—]\s*$",  # - 12 -
        r"^\s*\d+\s*$",  # Standalone number on line
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)
    
    return text


# =============================================================================
# Main Normalization API
# =============================================================================

def normalize_ocr_strict(text: str) -> str:
    """Strict OCR normalization for checksums and exact matching.
    
    Applies aggressive normalization suitable for:
    - Page content checksums
    - Duplicate detection
    - Identical content verification
    
    Args:
        text: Input text from OCR
        
    Returns:
        Strictly normalized text (lowercase, no citations, no page numbers)
    """
    config = OCRNormalizationConfig.strict()
    return _apply_normalization(text, config)


def normalize_ocr_compare(text: str) -> str:
    """Comparison OCR normalization for similarity matching.
    
    Applies normalization suitable for:
    - Text similarity comparison
    - Diff detection
    - Change classification
    
    Preserves semantic content while normalizing OCR artifacts.
    
    Args:
        text: Input text from OCR
        
    Returns:
        Normalized text for comparison (preserves case, citations)
    """
    config = OCRNormalizationConfig.compare()
    return _apply_normalization(text, config)


def normalize_ocr_text(text: str, config: Optional[OCRNormalizationConfig] = None) -> str:
    """Apply OCR normalization with custom configuration.
    
    Args:
        text: Input text from OCR
        config: Normalization configuration (defaults to compare mode)
        
    Returns:
        Normalized text according to configuration
    """
    if config is None:
        config = OCRNormalizationConfig.compare()
    return _apply_normalization(text, config)


def _apply_normalization(text: str, config: OCRNormalizationConfig) -> str:
    """Apply normalization according to configuration."""
    if not text:
        return ""
    
    # Unicode NFC normalization first
    result = unicodedata.normalize("NFC", text)
    
    # Ligatures
    if config.expand_ligatures:
        result = expand_ligatures(result)
    
    # Hyphenation (before whitespace normalization)
    if config.merge_hyphenated_words:
        result = merge_hyphenated_words(result)
    
    # Citations
    if config.normalize_citations:
        result = normalize_citations(result, strip=config.strip_citations)
    
    # Page numbers
    if config.strip_page_numbers:
        result = strip_page_numbers(result)
    
    # Quotes
    if config.normalize_quotes:
        result = normalize_quotes(result)
    
    # Dashes
    if config.normalize_dashes:
        result = normalize_dashes(result)
    
    # Diacritics (usually only for near-miss detection)
    if config.strip_diacritics:
        result = strip_diacritics(result)
    
    # Whitespace
    if config.collapse_whitespace or config.normalize_line_breaks:
        result = normalize_whitespace(
            result,
            soft_line_breaks=config.soft_line_break_equivalence,
        )
    
    # Case
    if config.lowercase:
        result = result.lower()
    
    return result


# =============================================================================
# Diacritics Near-Miss Detection
# =============================================================================

@dataclass
class DiacriticsAnalysis:
    """Analysis of diacritics differences between two texts."""
    
    # Count of diacritics-only differences
    diacritics_diffs: int = 0
    
    # Characters that differ only in diacritics
    diff_pairs: List[Tuple[str, str]] = field(default_factory=list)
    
    # Whether the difference is diacritics-only
    is_diacritics_only: bool = False
    
    # Severity classification
    severity: str = "none"  # "none", "low", "medium", "high"


def analyze_diacritics_difference(text_a: str, text_b: str) -> DiacriticsAnalysis:
    """Analyze if two texts differ only in diacritics.
    
    This is crucial for OCR where ė might be read as e, or ū as u.
    Such differences should be classified as low-severity OCR errors,
    not content changes.
    
    Args:
        text_a: First text
        text_b: Second text
        
    Returns:
        DiacriticsAnalysis with details about diacritics differences
    """
    analysis = DiacriticsAnalysis()
    
    # Normalize both texts
    norm_a = normalize_ocr_compare(text_a)
    norm_b = normalize_ocr_compare(text_b)
    
    # If normalized texts are identical, no diacritics issue
    if norm_a == norm_b:
        return analysis
    
    # Strip diacritics from both
    stripped_a = strip_diacritics(norm_a)
    stripped_b = strip_diacritics(norm_b)
    
    # If stripped versions are identical, it's diacritics-only
    if stripped_a == stripped_b:
        analysis.is_diacritics_only = True
        
        # Find specific differences
        for i, (c_a, c_b) in enumerate(zip(norm_a, norm_b)):
            if c_a != c_b:
                analysis.diacritics_diffs += 1
                analysis.diff_pairs.append((c_a, c_b))
        
        # Also check length differences
        if len(norm_a) != len(norm_b):
            analysis.diacritics_diffs += abs(len(norm_a) - len(norm_b))
        
        # Classify severity
        total_chars = max(len(norm_a), len(norm_b), 1)
        diff_ratio = analysis.diacritics_diffs / total_chars
        
        if diff_ratio < 0.01:
            analysis.severity = "low"
        elif diff_ratio < 0.05:
            analysis.severity = "medium"
        else:
            analysis.severity = "high"
    
    return analysis


# =============================================================================
# OCR Quality Classification
# =============================================================================

@dataclass
class OCRDiffClassification:
    """Classification of an OCR text difference."""
    
    # Primary classification
    category: str  # "content", "formatting", "ocr_noise", "diacritics"
    
    # Severity
    severity: str  # "high", "medium", "low", "none"
    
    # Confidence in classification
    confidence: float  # 0.0 - 1.0
    
    # Is this a phantom diff (should probably be filtered)?
    is_phantom: bool = False
    
    # Explanation
    reason: str = ""
    
    # Detailed analysis
    diacritics_analysis: Optional[DiacriticsAnalysis] = None


def classify_ocr_diff(
    text_a: str,
    text_b: str,
    is_ocr: bool = True,
) -> OCRDiffClassification:
    """Classify an OCR text difference.
    
    Determines whether a difference is:
    - Real content change
    - OCR noise (whitespace, punctuation artifacts)
    - Diacritics near-miss (ė vs e)
    - Formatting only
    
    Args:
        text_a: Original text
        text_b: Changed text
        is_ocr: Whether this is OCR-extracted text
        
    Returns:
        OCRDiffClassification with category and severity
    """
    # Handle empty cases
    if not text_a and not text_b:
        return OCRDiffClassification(
            category="none",
            severity="none",
            confidence=1.0,
            is_phantom=True,
            reason="Both texts empty",
        )
    
    if not text_a or not text_b:
        return OCRDiffClassification(
            category="content",
            severity="high",
            confidence=0.95,
            reason="Text added or deleted",
        )
    
    # Normalize for comparison
    norm_a = normalize_ocr_compare(text_a)
    norm_b = normalize_ocr_compare(text_b)
    
    # Exact match after normalization = whitespace/formatting only
    if norm_a == norm_b:
        return OCRDiffClassification(
            category="formatting",
            severity="low",
            confidence=0.9,
            is_phantom=is_ocr,  # Phantom if OCR (whitespace is unreliable)
            reason="Identical after OCR normalization (whitespace/punctuation noise)",
        )
    
    # Check for diacritics-only difference
    diacritics = analyze_diacritics_difference(text_a, text_b)
    if diacritics.is_diacritics_only:
        return OCRDiffClassification(
            category="diacritics",
            severity=diacritics.severity,
            confidence=0.85,
            is_phantom=is_ocr and diacritics.severity == "low",
            reason=f"Diacritics-only difference ({diacritics.diacritics_diffs} chars)",
            diacritics_analysis=diacritics,
        )
    
    # Check strict normalization (more aggressive)
    strict_a = normalize_ocr_strict(text_a)
    strict_b = normalize_ocr_strict(text_b)
    
    if strict_a == strict_b:
        return OCRDiffClassification(
            category="formatting",
            severity="low",
            confidence=0.85,
            is_phantom=is_ocr,
            reason="Identical after strict normalization (case/citation noise)",
        )
    
    # Calculate similarity to determine severity
    from rapidfuzz.fuzz import ratio
    similarity = ratio(norm_a, norm_b) / 100.0
    
    if similarity > 0.95:
        return OCRDiffClassification(
            category="ocr_noise",
            severity="low",
            confidence=0.8,
            is_phantom=is_ocr,
            reason=f"Very high similarity ({similarity:.2%}) - likely OCR noise",
        )
    elif similarity > 0.85:
        return OCRDiffClassification(
            category="content" if not is_ocr else "ocr_noise",
            severity="medium",
            confidence=0.7,
            is_phantom=is_ocr and similarity > 0.90,
            reason=f"High similarity ({similarity:.2%}) - minor change or OCR variation",
        )
    else:
        return OCRDiffClassification(
            category="content",
            severity="high",
            confidence=0.9,
            is_phantom=False,
            reason=f"Significant difference ({similarity:.2%})",
        )


# =============================================================================
# Batch Processing Utilities
# =============================================================================

def prefilter_identical_ocr(
    texts_a: List[str],
    texts_b: List[str],
) -> Tuple[List[int], List[int]]:
    """Prefilter to find identical text pairs after OCR normalization.
    
    Returns indices of pairs that are identical (can skip detailed diff).
    
    Args:
        texts_a: List of texts from document A
        texts_b: List of texts from document B
        
    Returns:
        Tuple of (identical_indices_a, different_indices_a)
    """
    identical = []
    different = []
    
    for i, (a, b) in enumerate(zip(texts_a, texts_b)):
        if normalize_ocr_strict(a) == normalize_ocr_strict(b):
            identical.append(i)
        else:
            different.append(i)
    
    return identical, different
