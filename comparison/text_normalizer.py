"""
Text Normalizer Module - Phase 2 Optimization

Implements two-text model for comparison:
- strict_text: For display/rendering (preserves formatting)
- compare_text: For matching (normalized for comparison)

Anti-phantom diff features:
- Whitespace collapse
- Hyphenation join
- Quote/dash normalization
- Case normalization (optional)
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Unicode Normalization Tables
# =============================================================================

# Quote variants to normalize
QUOTE_VARIANTS = {
    # Single quotes
    "\u2018": "'",  # LEFT SINGLE QUOTATION MARK
    "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK
    "\u201A": "'",  # SINGLE LOW-9 QUOTATION MARK
    "\u201B": "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    "\u2039": "'",  # SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    "\u203A": "'",  # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    "`": "'",       # GRAVE ACCENT
    # Double quotes
    "\u201C": '"',  # LEFT DOUBLE QUOTATION MARK
    "\u201D": '"',  # RIGHT DOUBLE QUOTATION MARK
    "\u201E": '"',  # DOUBLE LOW-9 QUOTATION MARK
    "\u201F": '"',  # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    "\u00AB": '"',  # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    "\u00BB": '"',  # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
}

# Dash/hyphen variants to normalize
DASH_VARIANTS = {
    "\u2010": "-",  # HYPHEN
    "\u2011": "-",  # NON-BREAKING HYPHEN
    "\u2012": "-",  # FIGURE DASH
    "\u2013": "-",  # EN DASH
    "\u2014": "-",  # EM DASH
    "\u2015": "-",  # HORIZONTAL BAR
    "\u2212": "-",  # MINUS SIGN
    "\uFE58": "-",  # SMALL EM DASH
    "\uFE63": "-",  # SMALL HYPHEN-MINUS
    "\uFF0D": "-",  # FULLWIDTH HYPHEN-MINUS
}

# Special whitespace characters
SPECIAL_SPACES = {
    "\u00A0": " ",  # NO-BREAK SPACE
    "\u1680": " ",  # OGHAM SPACE MARK
    "\u2000": " ",  # EN QUAD
    "\u2001": " ",  # EM QUAD
    "\u2002": " ",  # EN SPACE
    "\u2003": " ",  # EM SPACE
    "\u2004": " ",  # THREE-PER-EM SPACE
    "\u2005": " ",  # FOUR-PER-EM SPACE
    "\u2006": " ",  # SIX-PER-EM SPACE
    "\u2007": " ",  # FIGURE SPACE
    "\u2008": " ",  # PUNCTUATION SPACE
    "\u2009": " ",  # THIN SPACE
    "\u200A": " ",  # HAIR SPACE
    "\u202F": " ",  # NARROW NO-BREAK SPACE
    "\u205F": " ",  # MEDIUM MATHEMATICAL SPACE
    "\u3000": " ",  # IDEOGRAPHIC SPACE
}

# Zero-width characters to remove
ZERO_WIDTH_CHARS = {
    "\u200B",  # ZERO WIDTH SPACE
    "\u200C",  # ZERO WIDTH NON-JOINER
    "\u200D",  # ZERO WIDTH JOINER
    "\uFEFF",  # ZERO WIDTH NO-BREAK SPACE (BOM)
    "\u2060",  # WORD JOINER
    "\u00AD",  # SOFT HYPHEN
}

# Compile translation tables
_QUOTE_TABLE = str.maketrans(QUOTE_VARIANTS)
_DASH_TABLE = str.maketrans(DASH_VARIANTS)
_SPACE_TABLE = str.maketrans(SPECIAL_SPACES)

# Regex patterns
_ZERO_WIDTH_RE = re.compile(f"[{''.join(re.escape(c) for c in ZERO_WIDTH_CHARS)}]")
_WHITESPACE_RE = re.compile(r"\s+")
_HYPHENATION_RE = re.compile(r"-\s*[\n\r]+\s*")  # "word-\n" -> "word"
_MULTIPLE_DASHES_RE = re.compile(r"-{2,}")  # "--" or more -> "-"


@dataclass
class NormalizationConfig:
    """Configuration for text normalization.
    
    This defines what transformations are applied when converting
    strict_text to compare_text.
    """
    # Core normalization
    lowercase: bool = True
    nfc_normalize: bool = True  # Unicode NFC normalization
    
    # Whitespace handling
    collapse_whitespace: bool = True
    normalize_special_spaces: bool = True
    strip_zero_width: bool = True
    
    # Character normalization
    normalize_quotes: bool = True
    normalize_dashes: bool = True
    join_hyphenation: bool = True  # "word-\n" -> "word"
    
    # OCR-specific
    ocr_mode: bool = False  # Enable OCR-specific heuristics
    
    def to_dict(self) -> dict:
        """Export config to JSON-serializable dict."""
        return {
            "lowercase": self.lowercase,
            "nfc_normalize": self.nfc_normalize,
            "collapse_whitespace": self.collapse_whitespace,
            "normalize_special_spaces": self.normalize_special_spaces,
            "strip_zero_width": self.strip_zero_width,
            "normalize_quotes": self.normalize_quotes,
            "normalize_dashes": self.normalize_dashes,
            "join_hyphenation": self.join_hyphenation,
            "ocr_mode": self.ocr_mode,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "NormalizationConfig":
        """Create config from dict."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
    
    @classmethod
    def default_digital(cls) -> "NormalizationConfig":
        """Default config for digital PDFs."""
        return cls(ocr_mode=False)
    
    @classmethod
    def default_ocr(cls) -> "NormalizationConfig":
        """Default config for OCR-extracted PDFs."""
        return cls(ocr_mode=True, join_hyphenation=True)


@dataclass
class NormalizedText:
    """Container for dual-text representation.
    
    Attributes:
        strict_text: Original text with minimal normalization (for display)
        compare_text: Fully normalized text (for matching/comparison)
        fingerprint: SHA-1 hash of compare_text for quick equality checks
    """
    strict_text: str
    compare_text: str
    fingerprint: str
    
    @classmethod
    def from_text(
        cls,
        text: str,
        config: Optional[NormalizationConfig] = None,
    ) -> "NormalizedText":
        """Create NormalizedText from raw text.
        
        Args:
            text: Raw input text
            config: Normalization config (defaults to digital PDF settings)
        
        Returns:
            NormalizedText with both representations
        """
        if config is None:
            config = NormalizationConfig.default_digital()
        
        strict = normalize_strict(text)
        compare = normalize_compare(text, config)
        fp = compute_text_fingerprint(compare)
        
        return cls(strict_text=strict, compare_text=compare, fingerprint=fp)
    
    def matches(self, other: "NormalizedText") -> bool:
        """Quick equality check using fingerprint."""
        return self.fingerprint == other.fingerprint


def normalize_strict(text: str) -> str:
    """Minimal normalization for display/rendering.
    
    Only applies:
    - Unicode NFC normalization (for proper character composition)
    - Collapse multiple newlines to single
    - Strip leading/trailing whitespace
    
    Args:
        text: Raw input text
    
    Returns:
        Minimally normalized text suitable for display
    """
    if not text:
        return ""
    
    # NFC normalization (essential for Lithuanian and other accented chars)
    result = unicodedata.normalize("NFC", text)
    
    # Collapse multiple newlines but preserve single ones
    result = re.sub(r"\n{3,}", "\n\n", result)
    
    # Strip leading/trailing whitespace
    result = result.strip()
    
    return result


def normalize_compare(
    text: str,
    config: Optional[NormalizationConfig] = None,
) -> str:
    """Full normalization for comparison/matching.
    
    Applies comprehensive normalization to reduce false positive diffs.
    
    Args:
        text: Raw input text
        config: Normalization configuration
    
    Returns:
        Fully normalized text suitable for comparison
    """
    if not text:
        return ""
    
    if config is None:
        config = NormalizationConfig.default_digital()
    
    result = text
    
    # Step 1: Unicode NFC normalization
    if config.nfc_normalize:
        result = unicodedata.normalize("NFC", result)
    
    # Step 2: Handle hyphenation (must be before whitespace collapse)
    if config.join_hyphenation:
        result = _HYPHENATION_RE.sub("", result)
    
    # Step 3: Normalize special spaces
    if config.normalize_special_spaces:
        result = result.translate(_SPACE_TABLE)
    
    # Step 4: Remove zero-width characters
    if config.strip_zero_width:
        result = _ZERO_WIDTH_RE.sub("", result)
    
    # Step 5: Normalize quotes
    if config.normalize_quotes:
        result = result.translate(_QUOTE_TABLE)
    
    # Step 6: Normalize dashes
    if config.normalize_dashes:
        result = result.translate(_DASH_TABLE)
        # Collapse multiple dashes
        result = _MULTIPLE_DASHES_RE.sub("-", result)
        # Normalize space around dashes (OCR often adds spurious spaces)
        if config.ocr_mode:
            result = re.sub(r"\s*-\s*", "-", result)
    
    # Step 7: Collapse whitespace
    if config.collapse_whitespace:
        result = _WHITESPACE_RE.sub(" ", result)
    
    # Step 8: Lowercase
    if config.lowercase:
        result = result.lower()
    
    # Step 9: Strip
    result = result.strip()
    
    return result


def compute_text_fingerprint(text: str) -> str:
    """Compute SHA-1 fingerprint of normalized text.
    
    Used for quick equality checks at page level.
    
    Args:
        text: Normalized text
    
    Returns:
        SHA-1 hex digest (40 chars)
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def compute_page_fingerprint(lines: List[str], config: Optional[NormalizationConfig] = None) -> str:
    """Compute fingerprint for entire page content.
    
    Used for early termination - if page fingerprints match,
    skip detailed comparison.
    
    Args:
        lines: List of text lines on the page
        config: Normalization config
    
    Returns:
        SHA-1 hex digest of normalized page content
    """
    if config is None:
        config = NormalizationConfig.default_digital()
    
    # Normalize and join all lines
    normalized_lines = [normalize_compare(line, config) for line in lines]
    page_text = "\n".join(normalized_lines)
    
    return compute_text_fingerprint(page_text)


# =============================================================================
# Diff Classification Helpers
# =============================================================================

def is_whitespace_only_diff(text_a: str, text_b: str, config: Optional[NormalizationConfig] = None) -> bool:
    """Check if two texts differ only in whitespace.
    
    Args:
        text_a: First text
        text_b: Second text
        config: Normalization config
    
    Returns:
        True if texts are identical after whitespace normalization
    """
    if text_a == text_b:
        return False  # Identical, not a diff at all
    
    if config is None:
        config = NormalizationConfig.default_digital()
    
    norm_a = normalize_compare(text_a, config)
    norm_b = normalize_compare(text_b, config)
    
    return norm_a == norm_b


def is_formatting_only_diff(text_a: str, text_b: str, config: Optional[NormalizationConfig] = None) -> bool:
    """Check if two texts differ only in formatting (case, quotes, dashes).
    
    Args:
        text_a: First text
        text_b: Second text
        config: Normalization config
    
    Returns:
        True if texts are identical after formatting normalization
    """
    if text_a == text_b:
        return False  # Identical, not a diff at all
    
    if config is None:
        config = NormalizationConfig.default_digital()
    
    norm_a = normalize_compare(text_a, config)
    norm_b = normalize_compare(text_b, config)
    
    return norm_a == norm_b


def classify_text_diff(
    text_a: str,
    text_b: str,
    config: Optional[NormalizationConfig] = None,
) -> Tuple[bool, str]:
    """Classify the type of difference between two texts.
    
    Args:
        text_a: First text
        text_b: Second text
        config: Normalization config
    
    Returns:
        Tuple of (is_significant, diff_type)
        - is_significant: True if this is a real content change
        - diff_type: "identical", "whitespace_only", "formatting_only", "content"
    """
    if text_a == text_b:
        return False, "identical"
    
    if config is None:
        config = NormalizationConfig.default_digital()
    
    # Check if normalized versions are identical
    norm_a = normalize_compare(text_a, config)
    norm_b = normalize_compare(text_b, config)
    
    if norm_a == norm_b:
        # Check if it's whitespace-only by comparing with partial normalization
        ws_config = NormalizationConfig(
            lowercase=False,
            nfc_normalize=True,
            collapse_whitespace=True,
            normalize_special_spaces=True,
            strip_zero_width=True,
            normalize_quotes=False,
            normalize_dashes=False,
            join_hyphenation=config.join_hyphenation,
            ocr_mode=config.ocr_mode,
        )
        ws_norm_a = normalize_compare(text_a, ws_config)
        ws_norm_b = normalize_compare(text_b, ws_config)
        
        if ws_norm_a == ws_norm_b:
            return False, "whitespace_only"
        else:
            return False, "formatting_only"
    
    return True, "content"


# =============================================================================
# Batch Processing
# =============================================================================

def normalize_texts_batch(
    texts: List[str],
    config: Optional[NormalizationConfig] = None,
) -> List[NormalizedText]:
    """Normalize multiple texts in batch.
    
    Args:
        texts: List of raw texts
        config: Normalization config
    
    Returns:
        List of NormalizedText objects
    """
    if config is None:
        config = NormalizationConfig.default_digital()
    
    return [NormalizedText.from_text(text, config) for text in texts]


def deduplicate_for_encoding(
    texts: List[str],
    config: Optional[NormalizationConfig] = None,
) -> Tuple[List[str], Dict[int, int]]:
    """Deduplicate texts for efficient batch encoding.
    
    Returns unique normalized texts and a mapping from original indices
    to unique text indices.
    
    Args:
        texts: List of raw texts
        config: Normalization config
    
    Returns:
        Tuple of (unique_texts, index_mapping)
        - unique_texts: List of unique normalized texts (sorted for determinism)
        - index_mapping: Dict mapping original index -> unique text index
    """
    if config is None:
        config = NormalizationConfig.default_digital()
    
    # Normalize all texts
    normalized = [normalize_compare(t, config) for t in texts]
    
    # Build unique set with deterministic ordering
    unique_set = sorted(set(normalized))
    unique_to_idx = {text: idx for idx, text in enumerate(unique_set)}
    
    # Build index mapping
    index_mapping = {i: unique_to_idx[normalized[i]] for i in range(len(texts))}
    
    return unique_set, index_mapping


# =============================================================================
# Metrics Helpers
# =============================================================================

@dataclass
class NormalizationMetrics:
    """Metrics about normalization results for debugging/analysis."""
    total_texts: int = 0
    whitespace_only_diffs: int = 0
    formatting_only_diffs: int = 0
    content_diffs: int = 0
    identical_pairs: int = 0
    
    @property
    def whitespace_only_ratio(self) -> float:
        """Ratio of whitespace-only diffs to total."""
        total = self.whitespace_only_diffs + self.formatting_only_diffs + self.content_diffs
        return self.whitespace_only_diffs / max(1, total)
    
    @property
    def formatting_only_ratio(self) -> float:
        """Ratio of formatting-only diffs to total."""
        total = self.whitespace_only_diffs + self.formatting_only_diffs + self.content_diffs
        return self.formatting_only_diffs / max(1, total)
    
    def to_dict(self) -> dict:
        """Export metrics to dict."""
        return {
            "total_texts": self.total_texts,
            "whitespace_only_diffs": self.whitespace_only_diffs,
            "formatting_only_diffs": self.formatting_only_diffs,
            "content_diffs": self.content_diffs,
            "identical_pairs": self.identical_pairs,
            "whitespace_only_ratio": self.whitespace_only_ratio,
            "formatting_only_ratio": self.formatting_only_ratio,
        }


def compute_diff_metrics(
    text_pairs: List[Tuple[str, str]],
    config: Optional[NormalizationConfig] = None,
) -> NormalizationMetrics:
    """Compute normalization metrics for a list of text pairs.
    
    Args:
        text_pairs: List of (text_a, text_b) tuples
        config: Normalization config
    
    Returns:
        NormalizationMetrics with counts and ratios
    """
    metrics = NormalizationMetrics(total_texts=len(text_pairs))
    
    for text_a, text_b in text_pairs:
        is_significant, diff_type = classify_text_diff(text_a, text_b, config)
        
        if diff_type == "identical":
            metrics.identical_pairs += 1
        elif diff_type == "whitespace_only":
            metrics.whitespace_only_diffs += 1
        elif diff_type == "formatting_only":
            metrics.formatting_only_diffs += 1
        else:  # content
            metrics.content_diffs += 1
    
    return metrics
