"""Dual checksum system for page-level early termination.

Phase 2: Two checksum types for different document types:

1. Text Checksum (digital documents):
   - Hash of normalized text content
   - Fast, based on extracted text
   
2. Image Hash (scanned documents):  
   - Perceptual hash (dHash/pHash) of rendered page image
   - Handles OCR variance - if images are same, skip comparison
   - Uses imagehash library for robust visual hashing

This dramatically reduces phantom diffs for scanned documents where
OCR might produce slightly different text for visually identical pages.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from utils.ocr_normalizer import normalize_ocr_strict

if TYPE_CHECKING:
    from PIL import Image
    from comparison.models import PageData


# =============================================================================
# Checksum Types
# =============================================================================

class ChecksumType(Enum):
    """Type of checksum used for page comparison."""
    TEXT = "text"           # MD5 of normalized text
    IMAGE_DHASH = "dhash"   # Difference hash of rendered image
    IMAGE_PHASH = "phash"   # Perceptual hash of rendered image
    COMBINED = "combined"   # Both text and image


@dataclass
class PageChecksum:
    """Checksum data for a page."""
    
    page_num: int
    checksum_type: ChecksumType
    
    # Text checksum
    text_hash: Optional[str] = None
    text_char_count: int = 0
    
    # Image hashes
    image_dhash: Optional[str] = None
    image_phash: Optional[str] = None
    
    # Metadata
    is_ocr: bool = False
    render_dpi: int = 72
    
    def matches(self, other: "PageChecksum", tolerance: int = 8) -> bool:
        """Check if this checksum matches another.
        
        Args:
            other: Another PageChecksum to compare
            tolerance: Hamming distance tolerance for image hashes (0=exact)
            
        Returns:
            True if checksums match (pages are likely identical)
        """
        # Text hash match = definitely identical
        if self.text_hash and other.text_hash:
            if self.text_hash == other.text_hash:
                return True
        
        # For scanned docs, check image hashes
        if self.is_ocr or other.is_ocr:
            if self.image_dhash and other.image_dhash:
                distance = _hamming_distance(self.image_dhash, other.image_dhash)
                if distance <= tolerance:
                    return True
            
            if self.image_phash and other.image_phash:
                distance = _hamming_distance(self.image_phash, other.image_phash)
                if distance <= tolerance:
                    return True
        
        return False
    
    def similarity_score(self, other: "PageChecksum") -> float:
        """Compute similarity score between two checksums.
        
        Returns:
            Similarity score (0.0 - 1.0)
        """
        scores = []
        
        # Text hash (binary)
        if self.text_hash and other.text_hash:
            scores.append(1.0 if self.text_hash == other.text_hash else 0.0)
        
        # Image hash similarity
        if self.image_dhash and other.image_dhash:
            distance = _hamming_distance(self.image_dhash, other.image_dhash)
            # Max distance for 64-bit hash is 64
            scores.append(1.0 - distance / 64.0)
        
        if self.image_phash and other.image_phash:
            distance = _hamming_distance(self.image_phash, other.image_phash)
            scores.append(1.0 - distance / 64.0)
        
        return max(scores) if scores else 0.0


# =============================================================================
# Text Checksum Functions
# =============================================================================

def compute_text_checksum(page: "PageData") -> str:
    """Compute text-based checksum for a page.
    
    Uses strict OCR normalization to handle whitespace/punctuation variance.
    
    Args:
        page: PageData with extracted text blocks
        
    Returns:
        MD5 hash of normalized text content
    """
    # Collect all text blocks in reading order
    texts = []
    for block in (page.blocks or []):
        if block.text and block.text.strip():
            # Strict normalization for checksum
            normalized = normalize_ocr_strict(block.text.strip())
            if normalized:
                texts.append(normalized)
    
    # Join and hash
    content = "\n".join(texts)
    return hashlib.md5(content.encode('utf-8')).hexdigest()[:12]


def compute_text_checksum_from_string(text: str) -> str:
    """Compute text checksum from a raw string."""
    normalized = normalize_ocr_strict(text)
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:12]


# =============================================================================
# Image Hash Functions
# =============================================================================

def _hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two hex hash strings."""
    if len(hash1) != len(hash2):
        return 64  # Max distance
    
    # Convert hex to int
    try:
        int1 = int(hash1, 16)
        int2 = int(hash2, 16)
    except ValueError:
        return 64
    
    # Count differing bits
    xor = int1 ^ int2
    return bin(xor).count('1')


def compute_image_dhash(image: "Image.Image", hash_size: int = 8) -> str:
    """Compute difference hash (dHash) of an image.
    
    dHash is fast and good for detecting near-identical images.
    It compares adjacent pixels rather than absolute values.
    
    Args:
        image: PIL Image to hash
        hash_size: Size of hash (8 = 64-bit hash)
        
    Returns:
        Hexadecimal hash string
    """
    try:
        import imagehash
        dhash = imagehash.dhash(image, hash_size=hash_size)
        return str(dhash)
    except ImportError:
        # Fallback: simple implementation
        return _simple_dhash(image, hash_size)


def compute_image_phash(image: "Image.Image", hash_size: int = 8) -> str:
    """Compute perceptual hash (pHash) of an image.
    
    pHash is more robust to small changes but slower than dHash.
    Uses DCT (discrete cosine transform) for frequency analysis.
    
    Args:
        image: PIL Image to hash
        hash_size: Size of hash (8 = 64-bit hash)
        
    Returns:
        Hexadecimal hash string
    """
    try:
        import imagehash
        phash = imagehash.phash(image, hash_size=hash_size)
        return str(phash)
    except ImportError:
        # Fallback to dHash
        return _simple_dhash(image, hash_size)


def _simple_dhash(image: "Image.Image", hash_size: int = 8) -> str:
    """Simple dHash implementation without imagehash library."""
    # Resize to (hash_size + 1, hash_size)
    image = image.convert('L')  # Grayscale
    image = image.resize((hash_size + 1, hash_size), resample=1)  # BILINEAR
    
    pixels = list(image.getdata())
    
    # Compare adjacent pixels
    bits = []
    for row in range(hash_size):
        for col in range(hash_size):
            idx = row * (hash_size + 1) + col
            bits.append(1 if pixels[idx] < pixels[idx + 1] else 0)
    
    # Convert to hex
    hash_int = sum(bit << i for i, bit in enumerate(bits))
    return format(hash_int, f'0{hash_size * hash_size // 4}x')


def compute_image_hash_from_bytes(
    image_bytes: bytes,
    hash_type: str = "dhash",
) -> Optional[str]:
    """Compute image hash from raw bytes.
    
    Args:
        image_bytes: Raw image data (PNG, JPEG, etc.)
        hash_type: "dhash" or "phash"
        
    Returns:
        Hash string or None if failed
    """
    try:
        from PIL import Image
        import io
        
        image = Image.open(io.BytesIO(image_bytes))
        
        if hash_type == "phash":
            return compute_image_phash(image)
        else:
            return compute_image_dhash(image)
    except Exception:
        return None


# =============================================================================
# Page Rendering for Image Hash
# =============================================================================

def render_page_for_hash(
    pdf_path: str,
    page_num: int,
    dpi: int = 72,
) -> Optional["Image.Image"]:
    """Render a PDF page to image for hashing.
    
    Uses low DPI for speed - we only need structural similarity.
    
    Args:
        pdf_path: Path to PDF file
        page_num: 1-based page number
        dpi: Render resolution (72 is enough for hashing)
        
    Returns:
        PIL Image or None if failed
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image
        
        doc = fitz.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return None
        
        page = doc[page_num - 1]  # 0-based
        
        # Render at low DPI
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to PIL
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        doc.close()
        return img
    except Exception:
        return None


# =============================================================================
# Main Checksum API
# =============================================================================

def compute_page_checksum(
    page: "PageData",
    checksum_type: ChecksumType = ChecksumType.TEXT,
    pdf_path: Optional[str] = None,
    render_dpi: int = 72,
) -> PageChecksum:
    """Compute checksum(s) for a page.
    
    Args:
        page: PageData with extracted content
        checksum_type: Type of checksum to compute
        pdf_path: Path to PDF (required for image hash)
        render_dpi: DPI for rendering (for image hash)
        
    Returns:
        PageChecksum with computed hash(es)
    """
    # Detect if OCR was used
    md = page.metadata or {}
    extraction_method = str(md.get("extraction_method", ""))
    ocr_engine = str(md.get("ocr_engine_used", ""))
    is_ocr = (
        "ocr" in extraction_method.lower()
        or bool(ocr_engine)
    )
    
    result = PageChecksum(
        page_num=page.page_num,
        checksum_type=checksum_type,
        is_ocr=is_ocr,
        render_dpi=render_dpi,
    )
    
    # Always compute text hash
    result.text_hash = compute_text_checksum(page)
    result.text_char_count = sum(
        len(b.text or "") for b in (page.blocks or [])
    )
    
    # Compute image hash if requested and PDF available
    if checksum_type in (ChecksumType.IMAGE_DHASH, ChecksumType.IMAGE_PHASH, ChecksumType.COMBINED):
        if pdf_path:
            img = render_page_for_hash(pdf_path, page.page_num, render_dpi)
            if img:
                if checksum_type in (ChecksumType.IMAGE_DHASH, ChecksumType.COMBINED):
                    result.image_dhash = compute_image_dhash(img)
                if checksum_type in (ChecksumType.IMAGE_PHASH, ChecksumType.COMBINED):
                    result.image_phash = compute_image_phash(img)
    
    return result


def compute_checksums_for_document(
    pages: List["PageData"],
    pdf_path: Optional[str] = None,
    use_image_hash: bool = False,
    render_dpi: int = 72,
) -> Dict[int, PageChecksum]:
    """Compute checksums for all pages in a document.
    
    Args:
        pages: List of PageData
        pdf_path: Path to PDF (for image hashes)
        use_image_hash: Whether to compute image hashes
        render_dpi: DPI for rendering
        
    Returns:
        Dict mapping page_num to PageChecksum
    """
    checksum_type = (
        ChecksumType.COMBINED if use_image_hash 
        else ChecksumType.TEXT
    )
    
    return {
        page.page_num: compute_page_checksum(
            page,
            checksum_type=checksum_type,
            pdf_path=pdf_path,
            render_dpi=render_dpi,
        )
        for page in pages
    }


def find_identical_pages(
    checksums_a: Dict[int, PageChecksum],
    checksums_b: Dict[int, PageChecksum],
    alignment_map: Dict[int, Tuple[int, float]],
    tolerance: int = 8,
) -> List[int]:
    """Find pages that are identical based on checksums.
    
    Args:
        checksums_a: Checksums for document A
        checksums_b: Checksums for document B
        alignment_map: Page alignment (page_a -> (page_b, score))
        tolerance: Hamming distance tolerance for image hashes
        
    Returns:
        List of page numbers in A that are identical to their aligned page in B
    """
    identical = []
    
    for page_a, (page_b, _) in alignment_map.items():
        if page_a not in checksums_a or page_b not in checksums_b:
            continue
        
        cs_a = checksums_a[page_a]
        cs_b = checksums_b[page_b]
        
        if cs_a.matches(cs_b, tolerance=tolerance):
            identical.append(page_a)
    
    return identical


# =============================================================================
# Checksum Statistics
# =============================================================================

@dataclass
class ChecksumStats:
    """Statistics from checksum-based early termination."""
    
    total_pages: int = 0
    identical_by_text: int = 0
    identical_by_image: int = 0
    different: int = 0
    
    time_text_hash: float = 0.0
    time_image_hash: float = 0.0
    
    @property
    def identical_total(self) -> int:
        return self.identical_by_text + self.identical_by_image
    
    @property
    def skip_rate(self) -> float:
        return self.identical_total / max(1, self.total_pages)
    
    def to_dict(self) -> Dict:
        return {
            "total_pages": self.total_pages,
            "identical_by_text": self.identical_by_text,
            "identical_by_image": self.identical_by_image,
            "identical_total": self.identical_total,
            "different": self.different,
            "skip_rate": self.skip_rate,
            "time_text_hash": self.time_text_hash,
            "time_image_hash": self.time_image_hash,
        }
