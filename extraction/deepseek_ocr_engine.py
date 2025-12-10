"""DeepSeek-OCR engine for OCR processing with grounding support."""
from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from comparison.models import PageData, Style, TextBlock
from config.settings import settings
from utils.logging import logger


class DeepSeekOCR:
    """Wrapper for DeepSeek-OCR model with lazy loading."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model: Optional[object] = None
        self._processor: Optional[object] = None
        self._tokenizer: Optional[object] = None
        self._cuda_available: Optional[bool] = None
    
    def _load_model(self):
        """Lazy load the OCR model."""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoModel, AutoProcessor, AutoTokenizer
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for DeepSeek-OCR. "
                "Install via `pip install transformers torch`."
            ) from exc
        
        logger.info("Loading DeepSeek-OCR model from %s", self.model_path)
        try:
            # Determine device and attention implementation
            # DeepSeek-OCR's infer() method has CUDA operations hardcoded, which fail on non-CUDA devices
            # Store CUDA availability for later checks
            self._cuda_available = torch.cuda.is_available()
            use_cuda = self._cuda_available
            use_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
            
            # Enable MPS fallback to CPU for unsupported operations
            if not use_cuda and use_mps:
                import os
                os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
                logger.info("Enabled MPS fallback to CPU for unsupported operations")
            
            # Note: DeepSeek-OCR's infer() method may have CUDA operations, but we'll try anyway
            # with MPS fallback enabled. If it fails, we'll catch the error gracefully.
            if not use_cuda:
                logger.info(
                    "CUDA not available. Attempting OCR on CPU/MPS. "
                    "Some operations may fall back to CPU automatically."
                )
            
            # Load model without flash_attention_2 on non-CUDA devices to avoid CUDA errors
            load_kwargs = {"trust_remote_code": True}
            if not use_cuda:
                # Use default attention implementation (not flash_attention_2) for CPU/MPS
                load_kwargs["_attn_implementation"] = "eager"  # or "sdpa" if available
            
            # Try loading from local path first, fallback to HuggingFace
            if Path(self.model_path).exists():
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
                self._model = AutoModel.from_pretrained(self.model_path, **load_kwargs)
            else:
                # Use HuggingFace model name
                model_name = self.model_path if "/" in self.model_path else f"deepseek-ai/{self.model_path}"
                self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                self._model = AutoModel.from_pretrained(model_name, **load_kwargs)
            
            # Set to eval mode
            self._model.eval()
            # DeepSeek-OCR's infer() method has CUDA operations hardcoded, so always use CPU
            # on non-CUDA devices to avoid CUDA errors in infer()
            if use_cuda:
                self._model = self._model.cuda()
                logger.info("Using CUDA GPU for OCR")
            else:
                # Always use CPU for DeepSeek-OCR on non-CUDA devices
                # The infer() method has CUDA operations that fail even on MPS
                self._model = self._model.cpu()
                logger.info("Using CPU for OCR (DeepSeek-OCR infer() requires CPU on non-CUDA devices)")
        except Exception as exc:
            logger.warning("Failed to load DeepSeek-OCR, falling back to basic OCR: %s", exc)
            self._model = None
            self._processor = None
            self._tokenizer = None
    
    def recognize(self, image, target_size: Optional[Tuple[float, float]] = None) -> List[TextBlock]:
        """
        Run OCR on an image and return text blocks with bounding boxes.
        
        Args:
            image: Input image (PIL Image, numpy array, or PyMuPDF pixmap)
            target_size: Optional (width, height) to scale coordinates to (e.g. PDF points)
        """
        self._load_model()
        
        if self._model is None:
            # Fallback: return empty blocks if model not available
            logger.warning("OCR model not available, returning empty blocks")
            return []
        
        # Attempt OCR even without CUDA - MPS fallback should handle unsupported operations
        
        try:
            from PIL import Image
            import torch
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("PIL, torch, and numpy are required for OCR") from exc
        
        # Convert input to PIL Image
        if hasattr(image, "samples") and hasattr(image, "width") and hasattr(image, "height"):
            # PyMuPDF pixmap
            img = Image.frombytes("RGB", [image.width, image.height], image.samples)
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            # Already a PIL Image
            img = image
        else:
            # Try to convert to PIL Image
            try:
                img = Image.fromarray(np.array(image))
            except Exception:
                img = image
        
        # Process with DeepSeek-OCR
        try:
            import torch
            import tempfile
            from pathlib import Path
            
            # DeepSeek-OCR uses infer() method with image file path
            # This works on both CUDA and MPS (though MPS may be slower)
            if hasattr(self._model, 'infer') and self._tokenizer is not None:
                # Try using infer() method (preferred for DeepSeek-OCR)
                return self._recognize_with_infer(img, target_size)
            else:
                # Fallback to standard generation method
                return self._recognize_standard(img)
        except Exception as exc:
            logger.error("OCR processing failed: %s", exc)
            return []
    
    def _recognize_with_infer(self, img, target_size: Optional[Tuple[float, float]] = None) -> List[TextBlock]:
        """Run OCR using DeepSeek-OCR's infer() method (works on both CUDA and MPS)."""
        try:
            import tempfile
            from pathlib import Path
            
            # Create temporary file for image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                img.save(tmp_file.name, 'PNG')
                image_file = tmp_file.name
            
            # Prepare prompt - use grounding if enabled, otherwise simple OCR
            if settings.deepseek_grounding_enabled:
                prompt = "<image>\n<|grounding|>Convert the document to markdown. "
            else:
                prompt = "<image>\nFree OCR."
            
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as output_dir:
                try:
                    # Ensure model is on the correct device before calling infer()
                    import torch
                    current_device = next(self._model.parameters()).device
                    # Keep model on CPU if CUDA is not available (MPS fallback will handle operations)
                    if not torch.cuda.is_available() and current_device.type != 'cpu':
                        logger.debug("Moving model to CPU before infer() call (CUDA not available, using CPU with MPS fallback)")
                        self._model = self._model.cpu()
                    # Call infer method - MPS fallback should handle unsupported operations
                    result = self._model.infer(
                        tokenizer=self._tokenizer,
                        prompt=prompt,
                        image_file=image_file,
                        output_path=output_dir,
                        base_size=settings.deepseek_base_size,
                        image_size=settings.deepseek_image_size,
                        crop_mode=settings.deepseek_crop_mode,
                        save_results=True,
                        test_compress=False,
                    )
                    
                    # Parse result - infer() may return a string or dict
                    markdown_text = ""
                    if isinstance(result, str):
                        markdown_text = result
                    elif isinstance(result, dict):
                        markdown_text = result.get("text", result.get("markdown", ""))
                    else:
                        # Try to read from output directory
                        output_files = list(Path(output_dir).glob("*.md"))
                        if output_files:
                            markdown_text = output_files[0].read_text(encoding='utf-8')
                    
                    # If we got markdown with grounding, parse it
                    target_w, target_h = target_size if target_size else img.size
                    
                    if markdown_text and settings.deepseek_grounding_enabled:
                        text_blocks = self._parse_markdown_with_grounding(
                            markdown_text, 
                            img.size,
                            target_size=(target_w, target_h)
                        )
                    else:
                        # Parse as regular text
                        text_blocks = self._create_text_blocks_from_ocr(
                            markdown_text, 
                            img.size,
                            target_size=(target_w, target_h)
                        )
                    
                    # Clean up temporary image file
                    try:
                        Path(image_file).unlink()
                    except Exception:
                        pass
                    
                    return text_blocks
                except Exception as e:
                    # Handle CUDA errors - check for various CUDA-related error messages
                    error_msg = str(e)
                    logger.warning("DeepSeek inference failed with error: %s", error_msg) # Added debug log
                    is_cuda_error = (
                        "CUDA" in error_msg or 
                        "cuda" in error_msg.lower() or 
                        "Torch not compiled with CUDA" in error_msg or
                        "not compiled with CUDA" in error_msg
                    )
                    if is_cuda_error:
                        # Log CUDA error for this page - router will handle fallback
                        logger.warning(
                            "DeepSeek-OCR requires CUDA which is not available on this page. "
                            "Falling back to next engine."
                        )
                        # Clean up and return empty blocks
                        try:
                            Path(image_file).unlink()
                        except Exception:
                            pass
                        return []
                    else:
                        raise
        except Exception as exc:
            # Check if it's a CUDA error - if so, mark it and return empty blocks
            error_msg = str(exc)
            is_cuda_error = (
                "CUDA" in error_msg or 
                "cuda" in error_msg.lower() or 
                "Torch not compiled with CUDA" in error_msg or
                "not compiled with CUDA" in error_msg
            )
            if is_cuda_error:
                # Log CUDA error for this page - router will handle fallback
                logger.warning(
                    "DeepSeek-OCR requires CUDA which is not available on this page. "
                    "Falling back to next engine."
                )
                try:
                    Path(image_file).unlink()
                except Exception:
                    pass
                return []
            logger.warning("Infer method failed, falling back to standard: %s", exc)
            # Clean up temp file
            try:
                Path(image_file).unlink()
            except Exception:
                pass
            return self._recognize_standard(img)
    
    def _recognize_standard(self, img) -> List[TextBlock]:
        """Standard OCR recognition using infer() method (fallback when processor fails)."""
        # Use infer() method with simple OCR prompt instead of processor
        # This is more reliable across different devices
        try:
            import tempfile
            from pathlib import Path
            
            # Create temporary file for image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                img.save(tmp_file.name, 'PNG')
                image_file = tmp_file.name
            
            # Simple OCR prompt (no grounding)
            prompt = "<image>\nFree OCR."
            
            if self._tokenizer is None:
                logger.warning("Tokenizer not available for standard recognition")
                try:
                    Path(image_file).unlink()
                except Exception:
                    pass
                return []
            
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as output_dir:
                try:
                    # Ensure model is on CPU if CUDA is not available
                    import torch
                    import os
                    current_device = next(self._model.parameters()).device
                    if not torch.cuda.is_available() and current_device.type != 'cpu':
                        logger.debug("Moving model to CPU before infer() call in standard (CUDA not available)")
                        self._model = self._model.cpu()
                    # Try infer() method - MPS fallback should handle unsupported operations
                    result = self._model.infer(
                        tokenizer=self._tokenizer,
                        prompt=prompt,
                        image_file=image_file,
                        output_path=output_dir,
                        base_size=settings.deepseek_base_size,
                        image_size=settings.deepseek_image_size,
                        crop_mode=settings.deepseek_crop_mode,
                        save_results=True,
                        test_compress=False,
                    )
                    
                    # Parse result
                    ocr_text = ""
                    if isinstance(result, str):
                        ocr_text = result
                    elif isinstance(result, dict):
                        ocr_text = result.get("text", result.get("markdown", ""))
                    else:
                        # Try to read from output directory
                        output_files = list(Path(output_dir).glob("*.md"))
                        if not output_files:
                            output_files = list(Path(output_dir).glob("*.txt"))
                        if output_files:
                            ocr_text = output_files[0].read_text(encoding='utf-8')
                    
                    # Create text blocks from OCR output
                    text_blocks = self._create_text_blocks_from_ocr(ocr_text, img.size)
                    
                    # Clean up temporary image file
                    try:
                        Path(image_file).unlink()
                    except Exception:
                        pass
                    
                    return text_blocks
                except Exception as e:
                    # Handle CUDA errors - check for various CUDA-related error messages
                    error_msg = str(e)
                    is_cuda_error = (
                        "CUDA" in error_msg or 
                        "cuda" in error_msg.lower() or 
                        "Torch not compiled with CUDA" in error_msg or
                        "not compiled with CUDA" in error_msg
                    )
                    if is_cuda_error:
                        # Log CUDA error for this page - router will handle fallback
                        logger.warning(
                            "DeepSeek-OCR requires CUDA which is not available on this page. "
                            "Falling back to next engine."
                        )
                        # Clean up and return empty blocks
                        try:
                            Path(image_file).unlink()
                        except Exception:
                            pass
                        return []
                    else:
                        raise
        except Exception as exc:
            # Check if it's a CUDA error - if so, mark it and return empty blocks silently
            error_msg = str(exc)
            is_cuda_error = (
                "CUDA" in error_msg or 
                "cuda" in error_msg.lower() or 
                "Torch not compiled with CUDA" in error_msg or
                "not compiled with CUDA" in error_msg
            )
            if is_cuda_error:
                # Log CUDA error for this page - router will handle fallback
                logger.warning(
                    "DeepSeek-OCR requires CUDA which is not available on this page. "
                    "Falling back to next engine."
                )
                try:
                    Path(image_file).unlink()
                except Exception:
                    pass
                return []
            logger.error("Standard OCR recognition failed: %s", exc)
            # Clean up temp file
            try:
                Path(image_file).unlink()
            except Exception:
                pass
            return []
    
    def _parse_ocr_output_text(self, outputs) -> str:
        """Extract text string from model outputs."""
        try:
            if hasattr(outputs, 'generated_text'):
                return str(outputs.generated_text)
            elif hasattr(outputs, 'text'):
                return str(outputs.text)
            elif hasattr(outputs, 'logits'):
                # Decode from logits
                generated_ids = outputs.logits.argmax(dim=-1)
                if hasattr(self._processor, 'tokenizer'):
                    return self._processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        except Exception as exc:
            logger.debug("Error parsing OCR output text: %s", exc)
        return ""
    
    def _parse_markdown_with_grounding(
        self, 
        markdown_text: str, 
        image_size: Tuple[int, int],
        target_size: Tuple[float, float]
    ) -> List[TextBlock]:
        """
        Parse Markdown output with grounding information to extract structured text blocks.
        
        Grounding tokens in DeepSeek-OCR output are in format: <locX,Y,W,H> where coordinates
        are in [0-999] scale. We convert these to normalized [0-1] coordinates.
        """
        text_blocks = []
        
        if not markdown_text or not markdown_text.strip():
            return text_blocks
        
        img_width, img_height = image_size
        target_width, target_height = target_size
        
        # Parse grounding tokens: <locX,Y,W,H> format
        # Pattern matches <loc followed by numbers separated by commas
        grounding_pattern = re.compile(r'<loc(\d+),(\d+),(\d+),(\d+)>')
        
        # Split markdown into blocks by headings, paragraphs, tables, lists
        lines = markdown_text.split('\n')
        current_block_text = []
        current_bbox = None
        current_style = None
        block_type = "paragraph"
        
        for line in lines:
            line = line.strip()
            # Skip empty lines, but careful not to skip lines with only tokens if they matter
            # Actually, we should process tokens
            if not line:
                # Empty line - finalize current block if any
                if current_block_text:
                    text_blocks.append(self._create_block_from_text(
                        '\n'.join(current_block_text),
                        current_bbox,
                        current_style,
                        block_type,
                        img_width,
                        img_height,
                        target_width,
                        target_height,
                    ))
                    current_block_text = []
                    current_bbox = None
                    current_style = None
                continue
            
            # Check for heading
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                # Finalize previous block
                if current_block_text:
                    text_blocks.append(self._create_block_from_text(
                        '\n'.join(current_block_text),
                        current_bbox,
                        current_style,
                        block_type,
                        img_width,
                        img_height,
                        target_width,
                        target_height,
                    ))
                
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2)
                
                # Extract grounding if present
                bbox = self._extract_grounding_bbox(line, img_width, img_height)
                style = Style(bold=True, size=24.0 - (level * 2))
                
                text_blocks.append(self._create_block_from_text(
                    heading_text,
                    bbox,
                    style,
                    f"heading_h{level}",
                    img_width,
                    img_height,
                    target_width,
                    target_height,
                ))
                
                current_block_text = []
                current_bbox = None
                current_style = None
                block_type = "paragraph"
                continue
            
            # Check for table row
            if line.startswith('|') and line.endswith('|'):
                # Finalize previous block
                if current_block_text:
                    text_blocks.append(self._create_block_from_text(
                        '\n'.join(current_block_text),
                        current_bbox,
                        current_style,
                        block_type,
                        img_width,
                        img_height,
                        target_width,
                        target_height,
                    ))
                    current_block_text = []
                
                # Parse table row
                bbox = self._extract_grounding_bbox(line, img_width, img_height)
                text_blocks.append(self._create_block_from_text(
                    line,
                    bbox,
                    None,
                    "table_row",
                    img_width,
                    img_height,
                    target_width,
                    target_height,
                ))
                current_bbox = None
                continue
            
            # Check for list item
            list_match = re.match(r'^[-*+]\s+(.+)$', line)
            if list_match:
                if current_block_text and block_type != "list":
                    # Finalize previous paragraph
                    text_blocks.append(self._create_block_from_text(
                        '\n'.join(current_block_text),
                        current_bbox,
                        current_style,
                        block_type,
                        img_width,
                        img_height,
                        target_width,
                        target_height,
                    ))
                    current_block_text = []
                
                list_text = list_match.group(1)
                bbox = self._extract_grounding_bbox(line, img_width, img_height)
                text_blocks.append(self._create_block_from_text(
                    list_text,
                    bbox,
                    None,
                    "list_item",
                    img_width,
                    img_height,
                    target_width,
                    target_height,
                ))
                current_bbox = None
                continue
            
            # Regular paragraph line
            # Extract grounding from line if present
            line_bbox = self._extract_grounding_bbox(line, img_width, img_height)
            if line_bbox:
                current_bbox = line_bbox
            
            current_block_text.append(line)
        
        # Finalize last block
        if current_block_text:
            text_blocks.append(self._create_block_from_text(
                '\n'.join(current_block_text),
                current_bbox,
                block_type,
                img_width,
                img_height,
                target_width,
                target_height,
            ))
        
        logger.debug("Parsed %d text blocks from Markdown with grounding (%d chars)", 
                    len(text_blocks), len(markdown_text))
        return text_blocks
    
        return None
    
    def _extract_grounding_bbox(self, text: str, img_width: float, img_height: float) -> Optional[Dict[str, float]]:
        """
        Extract bounding box from grounding token in text.
        
        Supports formats:
        1. DeepSeek-VL-V2: <|det|>[[x1, y1, x2, y2]]<|/det|> (0-1000 scale)
        2. Legacy: <locX,Y,W,H> (0-999 scale)
        """
        # Try DeepSeek-VL-V2 format first
        v2_pattern = re.compile(r'<\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>')
        match = v2_pattern.search(text)
        
        if match:
            x1 = int(match.group(1))
            y1 = int(match.group(2))
            x2 = int(match.group(3))
            y2 = int(match.group(4))
            
            # Convert [0-1000] to normalized [0-1]
            x_norm = x1 / 1000.0
            y_norm = y1 / 1000.0
            w_norm = (x2 - x1) / 1000.0
            h_norm = (y2 - y1) / 1000.0
            
            return {
                "x": x_norm * img_width,
                "y": y_norm * img_height,
                "width": max(0, w_norm * img_width),
                "height": max(0, h_norm * img_height),
            }

        # Try legacy format
        grounding_pattern = re.compile(r'<loc(\d+),(\d+),(\d+),(\d+)>')
        match = grounding_pattern.search(text)
        
        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            w = int(match.group(3))
            h = int(match.group(4))
            
            # Convert from [0-999] scale to normalized [0-1]
            x_norm = x / 999.0
            y_norm = y / 999.0
            w_norm = w / 999.0
            h_norm = h / 999.0
            
            # Convert to absolute coordinates
            return {
                "x": x_norm * img_width,
                "y": y_norm * img_height,
                "width": w_norm * img_width,
                "height": h_norm * img_height,
            }
        
        return None
    
    def _create_block_from_text(
        self,
        text: str,
        bbox: Optional[Dict[str, float]],
        style: Optional[Style],
        block_type: str,
        img_width: float,
        img_height: float,
        target_width: float,
        target_height: float,
    ) -> TextBlock:
        """Create a TextBlock from parsed text, bbox, and style."""
        # Scale bbox to target dimensions (PDF points) if different from image dimensions
        scale_x = target_width / img_width if img_width > 0 else 1.0
        scale_y = target_height / img_height if img_height > 0 else 1.0
        
        # If no bbox provided, create a default one
        if bbox is None:
            bbox = {
                "x": 0.0,
                "y": 0.0,
                "width": float(target_width),
                "height": float(target_height),
            }
        else:
            # Scale the bbox
            bbox = {
                "x": bbox["x"] * scale_x,
                "y": bbox["y"] * scale_y,
                "width": bbox["width"] * scale_x,
                "height": bbox["height"] * scale_y,
            }
        
        # Remove grounding tokens from text
        # Remove <|ref|>...</|ref|> and <|det|>[[...]]<|/det|> and legacy <loc...>
        text_clean = re.sub(r'<\|ref\|>.*?<\|/ref\|>', '', text)
        text_clean = re.sub(r'<\|det\|>\[\[.*?\]\]<\|/det\|>', '', text_clean)
        text_clean = re.sub(r'<loc\d+,\d+,\d+,\d+>', '', text_clean).strip()
        
        # Determine bbox source
        bbox_source = "grounding" if settings.deepseek_grounding_enabled else "approx"
        
        block = TextBlock(
            text=text_clean,
            bbox=bbox,
            style=style,
            metadata={
                "ocr_engine": "deepseek",
                "bbox_source": bbox_source,
            }
        )
        
        return block
    
    def _create_text_blocks_from_ocr(
        self, 
        text: str, 
        image_size: tuple,
        target_size: Tuple[float, float]
    ) -> List[TextBlock]:
        """
        Create text blocks from OCR output text (fallback method without grounding).
        
        For now, creates blocks from paragraphs. In a full implementation,
        this would parse structured output with bounding boxes.
        """
        text_blocks = []
        
        if not text or not text.strip():
            return text_blocks
        
        img_width, img_height = image_size
        target_width, target_height = target_size
        
        # Try to parse as Markdown even without grounding
        if any(marker in text for marker in ['#', '|', '- ', '* ']):
            return self._parse_markdown_with_grounding(text, image_size, target_size)
        
        # Split text into paragraphs/lines for better block structure
        paragraphs = text.split('\n\n')
        
        if len(paragraphs) == 1:
            # Single block for all text
            text_blocks.append(TextBlock(
                text=text.strip(),
                bbox={
                    "x": 0.0,
                    "y": 0.0,
                    "width": float(target_width),
                    "height": float(target_height),
                },
                style=None,
                metadata={
                    "ocr_engine": "deepseek",
                    "bbox_source": "approx",
                }
            ))
        else:
            # Multiple blocks for paragraphs
            y_offset = 0.0
            line_height = img_height / max(len(paragraphs), 1)
            
            for para in paragraphs:
                if para.strip():
                    text_blocks.append(TextBlock(
                        text=para.strip(),
                        bbox={
                            "x": 0.0,
                            "y": y_offset * (target_height / img_height), # Scale offset
                            "width": float(target_width),
                            "height": line_height * (target_height / img_height), # Scale height
                        },
                        style=None,
                        metadata={
                            "ocr_engine": "deepseek",
                            "bbox_source": "approx",
                        }
                    ))
                    y_offset += line_height
        
        logger.debug("Created %d text blocks from OCR output (%d chars)", len(text_blocks), len(text))
        return text_blocks
    


# Global OCR instance for reuse
_ocr_instance: Optional[DeepSeekOCR] = None


def get_ocr_instance(model_path: str) -> DeepSeekOCR:
    """Get or create a global OCR instance."""
    global _ocr_instance
    if _ocr_instance is None or _ocr_instance.model_path != model_path:
        _ocr_instance = DeepSeekOCR(model_path)
    return _ocr_instance


def ocr_pdf(path: str | Path, model_path: str) -> List[PageData]:
    """Process a PDF through OCR and return PageData."""
    path = Path(path)
    logger.info("Running OCR on PDF: %s", path)
    import time
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for OCR rendering. Install via `pip install PyMuPDF`."
        ) from exc
    
    doc = fitz.open(path)
    ocr = get_ocr_instance(model_path)
    pages: List[PageData] = []

    for page in doc:
        # Render page at 150 DPI for faster processing (sufficient for OCR)
        # Previous 300 DPI + 2x matrix was overkill and caused slow performance
        render_start = time.time()
        pix = page.get_pixmap(dpi=150)
        render_time = time.time() - render_start
        logger.debug("Page %d rendered in %.2fs", page.number + 1, render_time)

        ocr_start = time.time()
        # Pass page dimensions for coordinate scaling
        text_blocks = ocr.recognize(pix, target_size=(page.rect.width, page.rect.height))
        ocr_time = time.time() - ocr_start
        logger.info("Page %d processed in %.2fs (%d blocks)", page.number + 1, ocr_time, len(text_blocks))
        
        # Extract structure metadata from blocks
        structure_metadata = _extract_structure_metadata(text_blocks)
        
        page_data = PageData(
            page_num=page.number + 1,
            width=page.rect.width,
            height=page.rect.height,
            blocks=text_blocks,
        )
        # Determine fallback reason if empty blocks
        fallback_reason = None
        if not text_blocks:
            fallback_reason = "empty_result"
        
        page_data.metadata = {
            "extraction_method": "ocr_deepseek",
            "ocr_engine_used": "deepseek",
            "dpi": 150,
            "render_time": render_time,
            "ocr_time": ocr_time,
            "has_grounding": settings.deepseek_grounding_enabled,
            **structure_metadata,
        }
        
        if fallback_reason:
            page_data.metadata["ocr_fallback_reason"] = fallback_reason
        pages.append(page_data)
    
    doc.close()
    logger.info("OCR processed %d pages", len(pages))
    return pages


def _extract_structure_metadata(text_blocks: List[TextBlock]) -> Dict:
    """Extract structural metadata from text blocks (headings, tables, lists)."""
    metadata = {
        "headings": [],
        "tables": [],
        "lists": [],
        "has_markdown_structure": False,
    }
    
    for idx, block in enumerate(text_blocks):
        # Check block type from text patterns or stored metadata
        text = block.text
        
        # Check for headings (already parsed in _parse_markdown_with_grounding)
        if text.startswith('#') or (block.style and block.style.bold and block.style.size and block.style.size > 14):
            level = 1
            if text.startswith('#'):
                level = len(text) - len(text.lstrip('#'))
            metadata["headings"].append({
                "index": idx,
                "level": level,
                "text": text.lstrip('#').strip(),
                "bbox": block.bbox,
            })
            metadata["has_markdown_structure"] = True
        
        # Check for table rows
        if '|' in text and text.count('|') >= 2:
            if "tables" not in metadata or not metadata["tables"]:
                metadata["tables"].append({
                    "start_index": idx,
                    "rows": [],
                })
            metadata["tables"][-1]["rows"].append({
                "index": idx,
                "text": text,
                "bbox": block.bbox,
            })
            metadata["has_markdown_structure"] = True
        
        # Check for list items
        if re.match(r'^[-*+]\s+', text):
            metadata["lists"].append({
                "index": idx,
                "text": re.sub(r'^[-*+]\s+', '', text),
                "bbox": block.bbox,
            })
            metadata["has_markdown_structure"] = True
    
    return metadata
