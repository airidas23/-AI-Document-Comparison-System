"""DeepSeek-OCR engine for OCR processing (stable version for Mac M-series).

Guardrails v1:
- Subprocess-based hard timeout (killable)
- Memory monitoring with soft/hard limits
- Academic-safe two-tier validation
- Mode presets for different extraction strategies
- Deterministic fallback support via GuardrailViolation exception
"""
from __future__ import annotations

import gc
import multiprocessing as mp
import os
import re
import tempfile
import threading
import time
import warnings
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from comparison.models import PageData, TextBlock
from utils.logging import logger


# =============================================================================
# Guardrail Data Structures
# =============================================================================

@dataclass
class GuardrailDiagnostics:
    """Diagnostics for guardrail violations - logged for debugging."""
    reason: str = ""
    elapsed_sec: float = 0.0
    rss_mb: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    sample_lines: List[str] = field(default_factory=list)
    top_tokens: List[Tuple[str, int]] = field(default_factory=list)
    repetition_ratio: float = 0.0


@dataclass
class GuardrailResult:
    """Result from guardrail-wrapped inference."""
    ok: bool
    blocks: Optional[List[Dict[str, Any]]] = None
    raw_text: str = ""
    warnings: List[str] = field(default_factory=list)
    diagnostics: Optional[GuardrailDiagnostics] = None
    engine_meta: Dict[str, Any] = field(default_factory=dict)


class GuardrailViolation(Exception):
    """Exception raised when guardrails are triggered (timeout, memory, rejected output).
    
    Caught by OCR router to trigger fallback to next engine.
    """
    def __init__(self, reason: str, diagnostics: Optional[GuardrailDiagnostics] = None):
        super().__init__(reason)
        self.reason = reason
        self.diagnostics = diagnostics


@contextmanager
def _suppress_expected_warnings():
    """Suppress expected warnings during DeepSeek-OCR model loading."""
    import logging
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*model of type deepseek_vl_v2.*")
        warnings.filterwarnings("ignore", message=".*not initialized from the model checkpoint.*")
        warnings.filterwarnings("ignore", message=".*attention layers.*RoPE embeddings.*")
        warnings.filterwarnings("ignore", message=".*position_embeddings.*")
        warnings.filterwarnings("ignore", message=".*TRAIN this model.*")
        
        try:
            import transformers
            prev_verbosity = transformers.logging.get_verbosity()
            transformers.logging.set_verbosity_error()
        except (ImportError, AttributeError):
            prev_verbosity = None
        
        loggers_to_quiet = [
            "transformers.modeling_utils",
            "transformers.configuration_utils", 
            "transformers.tokenization_utils_base",
        ]
        original_levels = {}
        for logger_name in loggers_to_quiet:
            try:
                lgr = logging.getLogger(logger_name)
                original_levels[logger_name] = lgr.level
                lgr.setLevel(logging.ERROR)
            except Exception:
                pass
        
        try:
            yield
        finally:
            if prev_verbosity is not None:
                try:
                    transformers.logging.set_verbosity(prev_verbosity)
                except Exception:
                    pass
            for logger_name, level in original_levels.items():
                try:
                    logging.getLogger(logger_name).setLevel(level)
                except Exception:
                    pass


class DeepSeekOCR:
    """Wrapper for DeepSeek-OCR model with lazy loading and grounding support.
    
    Guardrails v1:
    - Thread-safe via _processing_lock (sequential inference only)
    - Subprocess-based hard timeout with terminate/kill
    - Memory monitoring with soft/hard limits
    - Academic-safe two-tier output validation
    """

    # Class-level lock for sequential processing (no parallelism)
    _processing_lock = threading.Lock()

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model: Optional[object] = None
        self._tokenizer: Optional[object] = None
        self._device: Optional[str] = None
        self._output_dir = tempfile.TemporaryDirectory(prefix="deepseek_ocr_")
        self._output_dir_path = self._output_dir.name
        self._last_reject_reason: Optional[str] = None
        self._last_quality: Dict[str, float] = {}
        self._last_guardrail_diag: Optional[GuardrailDiagnostics] = None

    def _load_model(self) -> None:
        """Lazy load the DeepSeek model on best available device."""
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for DeepSeek-OCR."
            ) from exc

        # Device selection
        cuda_built = bool(getattr(getattr(torch, "backends", None), "cuda", None)) and torch.backends.cuda.is_built()
        cuda_available = cuda_built and torch.cuda.is_available()

        if cuda_available:
            device = torch.device("cuda")
            attn_impl_candidates = ["flash_attention_2", "eager"]
            dtype_candidates = [torch.bfloat16]
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            attn_impl_candidates = ["eager"]
            # MPS stability: prefer fp32.
            # DeepSeek-OCR MPS support historically had dtype edge-cases; fp32 is slower
            # but much more reliable on Apple Silicon.
            # However, fp32 consumes 2x memory, leading to OOM (SIGKILL -9).
            # We try bfloat16/float16 first for memory efficiency, falling back to float32.
            dtype_candidates = [torch.bfloat16, torch.float16, torch.float32]
        else:
            device = torch.device("cpu")
            attn_impl_candidates = ["eager"]
            dtype_candidates = [torch.float32]

        self._device = device.type
        logger.info("[DeepSeek] Using device: %s", self._device)

        # Resolve model path
        model_ref = self.model_path
        raw_path = Path(model_ref).expanduser()
        candidates = []
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            project_root = Path(__file__).resolve().parents[1]
            candidates.append(project_root / raw_path)
            candidates.append(Path.cwd() / raw_path)

        resolved_local: Optional[Path] = None
        for cand in candidates:
            if not cand.exists():
                continue
            has_config = (cand / "config.json").exists()
            has_weights = bool(list(cand.glob("*.safetensors"))) or bool(list(cand.glob("*.bin")))
            if cand.is_dir() and has_config and has_weights:
                resolved_local = cand
                break

        if resolved_local is not None:
            model_ref = str(resolved_local)
        else:
            if "/" not in model_ref:
                model_ref = "deepseek-ai/DeepSeek-OCR"
            if Path(model_ref).expanduser().exists() and "/" not in self.model_path:
                model_ref = "deepseek-ai/DeepSeek-OCR"

        logger.info("[DeepSeek] Loading model from %s", model_ref)

        with _suppress_expected_warnings():
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_ref,
                trust_remote_code=True,
                use_fast=True,
            )

            last_err: Optional[Exception] = None
            self._model = None
            for attn_impl in attn_impl_candidates:
                for dtype in dtype_candidates:
                    try:
                        self._model = AutoModel.from_pretrained(
                            model_ref,
                            trust_remote_code=True,
                            use_safetensors=True,
                            _attn_implementation=attn_impl,
                            torch_dtype=dtype,
                        )
                        if self._device == "mps":
                            import os
                            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
                        self._model = self._model.eval().to(device).to(dtype)
                        
                        # Validate model structure
                        base = self._model.model if hasattr(self._model, "model") else self._model
                        if not hasattr(base, "projector"):
                            raise RuntimeError("Model missing 'projector' attribute.")
                        
                        # Ensure sub-models are on correct device/dtype
                        if hasattr(base, 'sam_model') and base.sam_model is not None:
                            base.sam_model = base.sam_model.to(device).to(dtype)
                        if hasattr(base, 'vision_model') and base.vision_model is not None:
                            base.vision_model = base.vision_model.to(device).to(dtype)
                        if hasattr(base, 'projector') and base.projector is not None:
                            base.projector = base.projector.to(device).to(dtype)
                        last_err = None
                        break
                    except Exception as exc:
                        last_err = exc
                        self._model = None
                if self._model is not None:
                    break

        if self._model is None:
            raise RuntimeError(f"Failed to load DeepSeek-OCR on {self._device}: {last_err}")

        if hasattr(self._model, "generation_config"):
            gc = self._model.generation_config
            gc.do_sample = False
            gc.temperature = 1.0
            gc.top_p = 1.0
            gc.max_new_tokens = 1024 
            gc.repetition_penalty = 1.2

            if (getattr(gc, "pad_token_id", None) is None and 
                getattr(self._tokenizer, "eos_token_id", None) is not None):
                gc.pad_token_id = self._tokenizer.eos_token_id

        logger.info("[DeepSeek] Model loaded and ready on %s", self._device)

    def recognize(
        self, image, target_size: Optional[Tuple[float, float]] = None
    ) -> List[TextBlock]:
        """Run OCR on an image and return text blocks."""
        self._load_model()

        if self._model is None or self._tokenizer is None:
            logger.warning("[DeepSeek] Model not available, returning empty blocks.")
            return []

        try:
            from PIL import Image
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("PIL and numpy are required for OCR") from exc

        if hasattr(image, "samples") and hasattr(image, "width") and hasattr(image, "height"):
            img = Image.frombytes("RGB", [image.width, image.height], image.samples)
        elif isinstance(image, Image.Image):
            img = image
        else:
            img = Image.fromarray(np.array(image))

        try:
            return self._recognize_with_infer(img, target_size)
        except Exception as exc:
            logger.error("[DeepSeek] infer() failed: %s", exc)
            return []

    # =========================================================================
    # GUARDRAIL-ENABLED PUBLIC API
    # =========================================================================

    def recognize_page(
        self,
        image,
        page_index: int = 0,
        manifest_page: Optional[Dict] = None,
        target_size: Optional[Tuple[float, float]] = None,
    ) -> "GuardrailResult":
        """
        New guardrail-enabled entry point for single-page OCR.
        
        Uses mode presets, timeout/memory enforcement, and two-tier academic-safe validation.
        Falls back through modes on failure.
        
        Args:
            image: PIL Image or fitz Pixmap
            page_index: Page number (0-indexed) for logging
            manifest_page: Optional dict to update with engine_type on success
            target_size: Target page size in points
            
        Returns:
            GuardrailResult with ok=True on success, ok=False with diagnostics on failure
        """
        from config.settings import settings
        
        # Enforce sequential processing if configured
        if getattr(settings, "deepseek_disable_parallel", True):
            with self._processing_lock:
                return self._recognize_page_impl(image, page_index, manifest_page, target_size)
        else:
            return self._recognize_page_impl(image, page_index, manifest_page, target_size)

    def _recognize_page_impl(
        self,
        image,
        page_index: int,
        manifest_page: Optional[Dict],
        target_size: Optional[Tuple[float, float]],
    ) -> "GuardrailResult":
        """Internal implementation of recognize_page with mode loop and retries."""
        from config.settings import settings
        
        # Convert image to PIL
        try:
            import numpy as np
        except ImportError as exc:
            return GuardrailResult(
                ok=False,
                blocks=[],
                raw_text="",
                warnings=[],
                diagnostics=GuardrailDiagnostics(reason="import_error"),
                engine_meta={"engine_type": "deepseek", "error": str(exc)},
            )
        
        if hasattr(image, "samples") and hasattr(image, "width") and hasattr(image, "height"):
            img = Image.frombytes("RGB", [image.width, image.height], image.samples)
        elif isinstance(image, Image.Image):
            img = image
        else:
            img = Image.fromarray(np.array(image))
        
        # Get mode priorities and presets
        mode_priorities = getattr(settings, "deepseek_modes_priority", ["plain_text"])
        mode_presets = getattr(settings, "deepseek_mode_presets", {})
        max_retries = getattr(settings, "deepseek_max_retries", 1)
        
        all_warnings: List[str] = []
        last_diag: Optional[GuardrailDiagnostics] = None
        
        for mode_name in mode_priorities:
            mode_cfg = mode_presets.get(mode_name, {})
            logger.info("[DeepSeek] Page %d: trying mode '%s'", page_index, mode_name)
            
            for attempt in range(max_retries):
                try:
                    result = self._infer_with_guardrails(
                        img=img,
                        page_index=page_index,
                        mode_name=mode_name,
                        mode_cfg=mode_cfg,
                        target_size=target_size,
                    )
                    
                    # Success - update manifest and return
                    if manifest_page is not None:
                        manifest_page["engine_type"] = "deepseek"
                        manifest_page["deepseek_mode"] = mode_name
                    
                    result.warnings.extend(all_warnings)
                    self._last_guardrail_diag = result.diagnostics
                    return result
                    
                except GuardrailViolation as gv:
                    last_diag = gv.diagnostics
                    self._last_guardrail_diag = last_diag
                    all_warnings.append(f"mode={mode_name} attempt={attempt+1}: {gv.reason}")
                    logger.warning(
                        "[DeepSeek] Page %d, mode '%s', attempt %d failed: %s",
                        page_index, mode_name, attempt + 1, gv.reason
                    )
                    self._cleanup_after_failure()
                    
                except Exception as exc:
                    last_diag = GuardrailDiagnostics(reason=f"exception:{type(exc).__name__}")
                    self._last_guardrail_diag = last_diag
                    all_warnings.append(f"mode={mode_name} attempt={attempt+1}: {exc}")
                    logger.error(
                        "[DeepSeek] Page %d, mode '%s', attempt %d exception: %s",
                        page_index, mode_name, attempt + 1, exc
                    )
                    self._cleanup_after_failure()
        
        # All modes/retries exhausted
        return GuardrailResult(
            ok=False,
            blocks=[],
            raw_text="",
            warnings=all_warnings,
            diagnostics=last_diag or GuardrailDiagnostics(reason="all_modes_exhausted"),
            engine_meta={"engine_type": "deepseek", "status": "failed"},
        )

    def _infer_with_guardrails(
        self,
        img: Image.Image,
        page_index: int,
        mode_name: str,
        mode_cfg: Dict,
        target_size: Optional[Tuple[float, float]],
    ) -> "GuardrailResult":
        """
        Run inference with timeout and memory monitoring.
        
        Raises GuardrailViolation on timeout, memory limit, or validation failure.
        """
        from config.settings import settings
        
        timeout_sec = getattr(settings, "deepseek_timeout_sec_per_page", 60)
        memory_soft_mb = getattr(settings, "deepseek_memory_soft_mb", 4500)
        memory_hard_mb = getattr(settings, "deepseek_memory_hard_mb", 6000)
        use_hard_timeout = getattr(settings, "deepseek_hard_timeout", True)
        
        # Get mode-specific settings
        prompt_template = mode_cfg.get("prompt", "")
        base_size = mode_cfg.get("base_size", getattr(settings, "deepseek_base_size", 512))
        image_size = mode_cfg.get("image_size", base_size)
        use_grounding = mode_cfg.get("grounding", False)
        
        if use_hard_timeout:
            # Use subprocess for killable timeout
            raw_text, elapsed, peak_rss = self._infer_deepseek_in_subprocess(
                img=img,
                prompt_template=prompt_template,
                base_size=base_size,
                image_size=image_size,
                use_grounding=use_grounding,
                timeout_sec=timeout_sec,
                memory_hard_mb=memory_hard_mb,
            )
        else:
            # Soft timeout with memory monitoring thread
            raw_text, elapsed, peak_rss = self._infer_with_soft_timeout(
                img=img,
                prompt_template=prompt_template,
                base_size=base_size,
                image_size=image_size,
                use_grounding=use_grounding,
                timeout_sec=timeout_sec,
                memory_soft_mb=memory_soft_mb,
            )
        
        # Check memory limits
        if peak_rss > memory_hard_mb:
            diag = self._build_diagnostics(
                reason="memory_hard_limit",
                elapsed_sec=elapsed,
                rss_mb=peak_rss,
                raw_text=raw_text,
            )
            raise GuardrailViolation("memory_hard_limit", diag)
        
        if peak_rss > memory_soft_mb:
            logger.warning(
                "[DeepSeek] Page %d: memory soft limit exceeded (%.1f MB > %d MB)",
                page_index, peak_rss, memory_soft_mb
            )
        
        # Validate output quality (two-tier academic-safe)
        validation_result = self._validate_academic_safe(raw_text, page_index)
        
        if not validation_result["ok"]:
            diag = self._build_diagnostics(
                reason=validation_result["reason"],
                elapsed_sec=elapsed,
                rss_mb=peak_rss,
                raw_text=raw_text,
                metrics=validation_result.get("metrics"),
            )
            raise GuardrailViolation(validation_result["reason"], diag)
        
        # Build text blocks
        img_w, img_h = img.size
        target_w, target_h = target_size if target_size else (img_w, img_h)
        
        if use_grounding and ("<|det|>" in raw_text or "<|ref|>" in raw_text):
            blocks = self._parse_grounded_output(raw_text, (img_w, img_h), (target_w, target_h))
        else:
            blocks = self._create_text_blocks_from_ocr(raw_text, (img_w, img_h), (target_w, target_h))
        
        diag = self._build_diagnostics(
            reason="ok",
            elapsed_sec=elapsed,
            rss_mb=peak_rss,
            raw_text=raw_text,
        )
        
        return GuardrailResult(
            ok=True,
            blocks=blocks,
            raw_text=raw_text,
            warnings=validation_result.get("warnings", []),
            diagnostics=diag,
            engine_meta={
                "engine_type": "deepseek",
                "mode": mode_name,
                "elapsed_sec": elapsed,
                "peak_rss_mb": peak_rss,
            },
        )

    def _infer_deepseek_in_subprocess(
        self,
        img: Image.Image,
        prompt_template: str,
        base_size: int,
        image_size: int,
        use_grounding: bool,
        timeout_sec: int,
        memory_hard_mb: int,
    ) -> Tuple[str, float, float]:
        """
        Run DeepSeek inference with hard timeout.
        
        Uses persistent worker (model loaded once, reused) by default.
        Falls back to legacy subprocess-per-page when persistent worker is disabled.
        
        Returns:
            Tuple of (raw_text, elapsed_seconds, peak_rss_mb)
            
        Raises:
            GuardrailViolation on timeout or subprocess failure
        """
        from config.settings import settings
        import tempfile
        import uuid
        
        # Check if persistent worker is enabled (default: True)
        use_persistent_worker = getattr(settings, "deepseek_use_persistent_worker", True)
        
        # Save image to temp file (avoids pickle overhead)
        tmp_img_path = Path(tempfile.gettempdir()) / f"ds_guard_{uuid.uuid4().hex}.png"
        img.convert("RGB").save(tmp_img_path, "PNG")
        
        try:
            if use_persistent_worker:
                return self._infer_with_persistent_worker(
                    image_path=str(tmp_img_path),
                    prompt_template=prompt_template,
                    base_size=base_size,
                    image_size=image_size,
                    use_grounding=use_grounding,
                    timeout_sec=timeout_sec,
                    memory_hard_mb=memory_hard_mb,
                )
            else:
                return self._infer_with_legacy_subprocess(
                    img_path=tmp_img_path,
                    prompt_template=prompt_template,
                    base_size=base_size,
                    image_size=image_size,
                    use_grounding=use_grounding,
                    timeout_sec=timeout_sec,
                )
        finally:
            # Cleanup temp file
            if tmp_img_path.exists():
                try:
                    tmp_img_path.unlink()
                except Exception:
                    pass
    
    def _infer_with_persistent_worker(
        self,
        image_path: str,
        prompt_template: str,
        base_size: int,
        image_size: int,
        use_grounding: bool,
        timeout_sec: int,
        memory_hard_mb: int,
    ) -> Tuple[str, float, float]:
        """
        Run inference using persistent worker (WebUI-style model reuse).
        
        The worker keeps the model loaded across all pages, eliminating
        the ~30s model reload time per page.
        """
        from extraction.deepseek_persistent_worker import get_persistent_worker
        
        try:
            worker = get_persistent_worker(self.model_path)
            raw_text, elapsed, peak_rss = worker.infer(
                image_path=image_path,
                prompt_template=prompt_template,
                base_size=base_size,
                image_size=image_size,
                use_grounding=use_grounding,
                timeout_sec=timeout_sec,
                memory_hard_mb=memory_hard_mb,
            )
            return raw_text, elapsed, peak_rss
            
        except TimeoutError as e:
            diag = GuardrailDiagnostics(
                reason="persistent_worker_timeout",
                elapsed_sec=timeout_sec,
            )
            raise GuardrailViolation("persistent_worker_timeout", diag) from e
            
        except RuntimeError as e:
            diag = GuardrailDiagnostics(
                reason=f"persistent_worker_error:{str(e)[:50]}",
            )
            raise GuardrailViolation("persistent_worker_error", diag) from e
    
    def _infer_with_legacy_subprocess(
        self,
        img_path: Path,
        prompt_template: str,
        base_size: int,
        image_size: int,
        use_grounding: bool,
        timeout_sec: int,
    ) -> Tuple[str, float, float]:
        """
        Legacy subprocess-per-page inference (deprecated).
        
        Spawns a new subprocess for each page, reloading the model each time.
        This is slow but provides maximum isolation.
        """
        import pickle
        import uuid
        
        tmp_result_path = Path(tempfile.gettempdir()) / f"ds_result_{uuid.uuid4().hex}.pkl"
        
        try:
            ctx = mp.get_context("spawn")
            
            subprocess_args = {
                "model_path": self.model_path,
                "image_path": str(img_path),
                "result_path": str(tmp_result_path),
                "prompt_template": prompt_template,
                "base_size": base_size,
                "image_size": image_size,
                "use_grounding": use_grounding,
            }
            
            proc = ctx.Process(
                target=_subprocess_infer_worker,
                args=(subprocess_args,),
            )
            
            start_time = time.time()
            proc.start()
            proc.join(timeout=timeout_sec)
            elapsed = time.time() - start_time
            
            if proc.is_alive():
                logger.warning("[DeepSeek] Legacy subprocess timeout after %.1fs, terminating", elapsed)
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2)
                
                diag = GuardrailDiagnostics(
                    reason="subprocess_timeout",
                    elapsed_sec=elapsed,
                )
                raise GuardrailViolation("subprocess_timeout", diag)
            
            if proc.exitcode != 0:
                diag = GuardrailDiagnostics(
                    reason=f"subprocess_exit_{proc.exitcode}",
                    elapsed_sec=elapsed,
                )
                raise GuardrailViolation(f"subprocess_exit_{proc.exitcode}", diag)
            
            if not tmp_result_path.exists():
                diag = GuardrailDiagnostics(
                    reason="subprocess_no_result",
                    elapsed_sec=elapsed,
                )
                raise GuardrailViolation("subprocess_no_result", diag)
            
            with open(tmp_result_path, "rb") as f:
                result = pickle.load(f)
            
            if "error" in result:
                diag = GuardrailDiagnostics(
                    reason=f"subprocess_error:{result['error'][:50]}",
                    elapsed_sec=elapsed,
                )
                raise GuardrailViolation("subprocess_error", diag)
            
            return result["text"], elapsed, result.get("peak_rss_mb", 0.0)
            
        finally:
            if tmp_result_path.exists():
                try:
                    tmp_result_path.unlink()
                except Exception:
                    pass


    def _infer_with_soft_timeout(
        self,
        img: Image.Image,
        prompt_template: str,
        base_size: int,
        image_size: int,
        use_grounding: bool,
        timeout_sec: int,
        memory_soft_mb: int,
    ) -> Tuple[str, float, float]:
        """
        Run inference with soft timeout (warning only, no kill).
        
        Uses a background thread to monitor memory usage.
        """
        import psutil
        
        self._load_model()
        
        if self._model is None or self._tokenizer is None:
            raise GuardrailViolation(
                "model_not_loaded",
                GuardrailDiagnostics(reason="model_not_loaded"),
            )
        
        # Memory monitoring
        peak_rss = [psutil.Process().memory_info().rss / (1024 * 1024)]
        stop_monitor = threading.Event()
        
        def memory_monitor():
            while not stop_monitor.is_set():
                current_rss = psutil.Process().memory_info().rss / (1024 * 1024)
                peak_rss[0] = max(peak_rss[0], current_rss)
                time.sleep(0.5)
        
        monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
        monitor_thread.start()
        
        start_time = time.time()
        
        try:
            # Build prompt
            prompt = "<image>\n"
            if use_grounding:
                prompt += "<|grounding|>"
            if prompt_template:
                prompt += prompt_template
            else:
                prompt += "Perform OCR and return the result in reading order."
            
            # Save temp image
            tmp_path = Path(tempfile.gettempdir()) / f"deepseek_{os.getpid()}.png"
            img.convert("RGB").save(tmp_path, "PNG")
            
            try:
                output_path = getattr(self, "_output_dir_path", None) or "deepseek_out"
                Path(output_path).mkdir(parents=True, exist_ok=True)
                
                result = self._model.infer(
                    tokenizer=self._tokenizer,
                    prompt=prompt,
                    image_file=str(tmp_path),
                    output_path=output_path,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=False,
                    save_results=False,
                    test_compress=False,
                    eval_mode=True,
                )
                
                elapsed = time.time() - start_time
                
                # Check soft timeout (warning only)
                if elapsed > timeout_sec:
                    logger.warning(
                        "[DeepSeek] Soft timeout exceeded (%.1fs > %ds)",
                        elapsed, timeout_sec
                    )
                
                raw_text = result[0] if result else ""
                return raw_text, elapsed, peak_rss[0]
                
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
                        
        finally:
            stop_monitor.set()
            monitor_thread.join(timeout=1)

    def _validate_academic_safe(self, text: str, page_index: int) -> Dict:
        """
        Two-tier academic-safe validation.
        
        Tier 1: Hard reject for true garbage (very low alnum, very short, repetition loops)
        Tier 2: For URL-heavy content, strip header/footer zones and re-check
        
        Returns dict with keys:
            - ok: bool
            - reason: str
            - warnings: List[str]
            - metrics: Dict (optional)
        """
        from config.settings import settings
        
        min_chars = getattr(settings, "deepseek_academic_min_chars", 200)
        min_alnum = getattr(settings, "deepseek_academic_min_alnum", 0.40)
        max_url_ratio = getattr(settings, "deepseek_academic_max_url_ratio", 0.30)
        max_repetition = getattr(settings, "deepseek_academic_max_repetition", 0.50)
        
        warnings: List[str] = []
        
        if not text or len(text.strip()) == 0:
            return {"ok": False, "reason": "empty_output", "warnings": warnings}
        
        metrics = self._compute_text_metrics(text)
        
        # Tier 1: Hard reject for true garbage
        if metrics["char_count"] < min_chars * 0.5:  # Very short - definite fail
            return {"ok": False, "reason": "tier1_too_short", "warnings": warnings, "metrics": metrics}
        
        if metrics["alnum_ratio"] < min_alnum * 0.5:  # Very low alnum - definite fail
            return {"ok": False, "reason": "tier1_low_alnum", "warnings": warnings, "metrics": metrics}
        
        if metrics["repetition_ratio"] > max_repetition * 1.5:  # Severe repetition
            return {"ok": False, "reason": "tier1_repetition_loop", "warnings": warnings, "metrics": metrics}
        
        # Tier 2: Check if content is borderline but potentially academic
        is_borderline = (
            metrics["char_count"] < min_chars or
            metrics["alnum_ratio"] < min_alnum or
            metrics["url_ratio"] > max_url_ratio or
            metrics["repetition_ratio"] > max_repetition
        )
        
        if is_borderline:
            # Try stripping header/footer zones (often contain URLs/page numbers)
            stripped_text = self._strip_header_footer(text)
            stripped_metrics = self._compute_text_metrics(stripped_text)
            
            # Re-evaluate with stripped content
            if stripped_metrics["char_count"] >= min_chars:
                if stripped_metrics["alnum_ratio"] >= min_alnum:
                    if stripped_metrics["url_ratio"] <= max_url_ratio:
                        if stripped_metrics["repetition_ratio"] <= max_repetition:
                            warnings.append(
                                f"tier2_accepted_after_strip: url_ratio {metrics['url_ratio']:.2f} -> {stripped_metrics['url_ratio']:.2f}"
                            )
                            logger.info(
                                "[DeepSeek] Page %d: Tier 2 accept after header/footer strip",
                                page_index
                            )
                            return {"ok": True, "reason": "tier2_accepted", "warnings": warnings, "metrics": stripped_metrics}
            
            # Determine specific rejection reason
            if metrics["url_ratio"] > max_url_ratio:
                return {"ok": False, "reason": "tier2_url_heavy", "warnings": warnings, "metrics": metrics}
            if metrics["repetition_ratio"] > max_repetition:
                return {"ok": False, "reason": "tier2_repetition", "warnings": warnings, "metrics": metrics}
            if metrics["alnum_ratio"] < min_alnum:
                return {"ok": False, "reason": "tier2_low_alnum", "warnings": warnings, "metrics": metrics}
            return {"ok": False, "reason": "tier2_quality_fail", "warnings": warnings, "metrics": metrics}
        
        # All checks passed
        return {"ok": True, "reason": "ok", "warnings": warnings, "metrics": metrics}

    def _compute_text_metrics(self, text: str) -> Dict[str, float]:
        """Compute quality metrics for validation."""
        if not text:
            return {
                "char_count": 0,
                "alnum_ratio": 0.0,
                "url_ratio": 0.0,
                "repetition_ratio": 0.0,
                "nonprintable_ratio": 0.0,
            }
        
        n = len(text)
        alnum = sum(1 for c in text if c.isalnum())
        nonprintable = sum(1 for c in text if not c.isprintable() and c not in "\n\r\t")
        
        # URL detection
        tokens = text.split()
        url_tokens = sum(
            1 for t in tokens
            if any(pat in t.lower() for pat in ["http://", "https://", "www.", ".com", ".org", ".edu", ".net"])
        )
        url_ratio = url_tokens / max(1, len(tokens))
        
        # Repetition detection
        repetition_ratio = self._compute_repetition_ratio(text)
        
        return {
            "char_count": n,
            "alnum_ratio": alnum / max(1, n),
            "url_ratio": url_ratio,
            "repetition_ratio": repetition_ratio,
            "nonprintable_ratio": nonprintable / max(1, n),
        }

    def _compute_repetition_ratio(self, text: str) -> float:
        """Compute ratio of most repeated token to total tokens."""
        words = text.split()
        if len(words) < 5:
            return 0.0
        
        counts = Counter(w.lower().strip(".,;:!?()[]{}") for w in words if w.strip())
        if not counts:
            return 0.0
        
        max_count = max(counts.values())
        return max_count / len(words)

    def _strip_header_footer(self, text: str, header_lines: int = 3, footer_lines: int = 3) -> str:
        """
        Strip header and footer lines which often contain URLs, page numbers, etc.
        
        Args:
            text: Full OCR text
            header_lines: Number of lines to strip from top
            footer_lines: Number of lines to strip from bottom
        """
        lines = text.split("\n")
        if len(lines) <= header_lines + footer_lines + 2:
            # Too short to strip, return as-is
            return text
        
        return "\n".join(lines[header_lines:-footer_lines])

    def _build_diagnostics(
        self,
        reason: str,
        elapsed_sec: float = 0.0,
        rss_mb: float = 0.0,
        raw_text: str = "",
        metrics: Optional[Dict[str, float]] = None,
    ) -> "GuardrailDiagnostics":
        """Build diagnostics object from inference results."""
        lines = raw_text.split("\n") if raw_text else []
        sample_lines = lines[:5]
        
        # Compute top tokens for debugging
        words = raw_text.split() if raw_text else []
        counts = Counter(w.lower().strip(".,;:!?()[]{}") for w in words if len(w) > 2)
        top_tokens = dict(counts.most_common(10))
        
        return GuardrailDiagnostics(
            reason=reason,
            elapsed_sec=elapsed_sec,
            rss_mb=rss_mb,
            metrics=metrics,
            sample_lines=sample_lines,
            top_tokens=top_tokens,
            repetition_ratio=self._compute_repetition_ratio(raw_text) if raw_text else 0.0,
        )

    def _cleanup_after_failure(self):
        """Clean up resources after a failed inference attempt."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # MPS cleanup is limited, but trigger GC
                pass
        except Exception:
            pass

    # =========================================================================
    # ORIGINAL METHODS (preserved for backward compatibility)
    # =========================================================================

    def _recognize_with_infer(
        self, img: Image.Image, target_size: Optional[Tuple[float, float]] = None
    ) -> List[TextBlock]:
        """Call DeepSeek's infer() and parse output."""
        if not hasattr(self._model, "infer"):
            raise RuntimeError("DeepSeek model has no 'infer' method")

        from config.settings import settings
        import torch
        import uuid

        fmt = getattr(settings, "deepseek_tmp_image_format", "png").lower()
        if fmt not in ("png", "jpeg", "jpg"):
            fmt = "png"

        suffix = ".png" if fmt == "png" else ".jpg"
        tmp_path = Path(tempfile.gettempdir()) / f"deepseek_{uuid.uuid4().hex}{suffix}"
        img_to_save = img.convert("RGB")
        if fmt == "png":
            img_to_save.save(tmp_path, "PNG", optimize=True)
        else:
            q = int(getattr(settings, "deepseek_tmp_jpeg_quality", 92))
            img_to_save.save(tmp_path, "JPEG", quality=q, optimize=True)

        try:
            prompt_base = (
                "Perform OCR and return the result in reading order. "
                "If grounding is enabled, output one line per box using the format: "
                "<|det|>[[x1,y1,x2,y2]]<|ref|>LINE_TEXT<|/ref|>. "
                "Keep boxes tight around each line."
            )

            output_path = getattr(self, "_output_dir_path", None) or "deepseek_out"
            if output_path == "deepseek_out":
                Path(output_path).mkdir(parents=True, exist_ok=True)

            def _build_prompt(use_grounding: bool) -> str:
                prompt = "<image>\n"
                if use_grounding:
                    prompt += "<|grounding|>"
                return prompt + prompt_base

            def _run_infer(
                base_size: int, image_size: int, crop_mode: bool, prompt: str
            ):
                return self._model.infer(
                    tokenizer=self._tokenizer,
                    prompt=prompt,
                    image_file=str(tmp_path),
                    output_path=output_path,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    save_results=False,
                    test_compress=False,
                    eval_mode=True,
                )

            base_cfg = int(getattr(settings, "deepseek_base_size", 512))
            img_cfg = int(getattr(settings, "deepseek_image_size", base_cfg))
            crop_cfg = bool(getattr(settings, "deepseek_crop_mode", False))

            if crop_cfg:
                logger.warning("[DeepSeek] deepseek_crop_mode=True is very slow on MPS.")

            retry_img = int(getattr(settings, "deepseek_retry_image_size", 768))
            grounding_enabled = bool(getattr(settings, "deepseek_grounding_enabled", False))
            grounding_precision_only = bool(
                getattr(settings, "deepseek_grounding_precision_only", True)
            )
            quick_crop = False if grounding_precision_only else crop_cfg
            attempts = [
                (base_cfg, img_cfg, quick_crop, "quick"),
                (base_cfg, max(img_cfg, retry_img), False, "retry_bigger_image"),
            ]
            if crop_cfg:
                attempts.append(
                    (base_cfg, max(img_cfg, retry_img), True, "retry_crop")
                )

            result = None
            text = ""
            last_exc: Optional[Exception] = None
            use_grounding_final = False
            self._last_reject_reason = None
            self._last_quality = {}

            with torch.inference_mode():
                for base_size, image_size, crop_mode, tag in attempts:
                    use_grounding = grounding_enabled and (
                        not grounding_precision_only or crop_mode
                    )
                    prompt = _build_prompt(use_grounding)
                    try:
                        logger.info(
                            "[DeepSeek] infer() attempt=%s base_size=%s image_size=%s crop_mode=%s grounding=%s",
                            tag,
                            base_size,
                            image_size,
                            crop_mode,
                            use_grounding,
                        )
                        result = _run_infer(base_size, image_size, crop_mode, prompt)
                        last_exc = None
                    except Exception as exc:
                        last_exc = exc
                        msg = str(exc)
                        known = ("must match the size" in msg or "masked_scatter" in msg or "same dtypes" in msg)
                        logger.warning("[DeepSeek] infer() attempt=%s failed (%s)", tag, msg)
                        if not known:
                            raise
                        continue

                    if result is None:
                        text = ""
                    elif isinstance(result, dict):
                        text = result.get("markdown") or result.get("text") or ""
                    elif isinstance(result, str):
                        text = result
                    else:
                        text = str(result)

                    text = self._fix_repetition_loops(text)
                    ok, reason, q = self._is_sane_output(text)
                    self._last_quality = q
                    if ok:
                        self._last_reject_reason = None
                        use_grounding_final = use_grounding
                        break
                    self._last_reject_reason = reason
                    logger.warning(
                        "[DeepSeek] OCR output rejected (attempt=%s): %s metrics=%s",
                        tag,
                        reason,
                        q,
                    )
                    text = ""
                    result = None

            if not text:
                if last_exc:
                    logger.error("[DeepSeek] All attempts failed. Last error: %s", last_exc)
                return []

        except Exception as exc:
            logger.error("[DeepSeek] infer() failed: %s", exc)
            return []
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        img_w, img_h = img.size
        if target_size is None:
            target_w, target_h = img_w, img_h
        else:
            target_w, target_h = target_size

        # Use grounding parser if enabled and tokens are present
        if use_grounding_final and ("<|det|>" in text or "<|ref|>" in text):
            return self._parse_grounded_output(text, (img_w, img_h), (target_w, target_h))
        else:
            return self._create_text_blocks_from_ocr(text, (img_w, img_h), (target_w, target_h))

    def _parse_grounded_output(
        self,
        text: str,
        image_size: Tuple[int, int],
        target_size: Tuple[float, float],
    ) -> List[TextBlock]:
        """
        Parse DeepSeek output with grounding tokens (<|det|>, <|ref|>).
        Splits multi-line blocks into individual lines for better granularity.
        """
        blocks: List[TextBlock] = []
        if not text:
            return blocks

        img_w, img_h = image_size
        page_w, page_h = target_size

        # DeepSeek grounding coordinates can be either normalized (0..1000-ish)
        # or pixel-based. Detect dynamically.
        def _coord_to_page_scale(max_coord: int) -> Tuple[float, float, str]:
            if max_coord <= 1100:
                return (page_w / 1000.0, page_h / 1000.0, "norm1000")
            if img_w > 0 and img_h > 0:
                return (page_w / float(img_w), page_h / float(img_h), "pixels")
            return (page_w / 1000.0, page_h / 1000.0, "fallback")

        # Regex for grounded blocks: <|det|> [[x1, y1, x2, y2]] <|ref|> Content <|/ref|>
        # Note: The format might vary slightly, but usually it's:
        # <|det|>[[x1,y1,x2,y2]]<|ref|>Content<|/ref|>
        # or sometimes just content with det at start.
        # DeepSeek-VL2 grounding format: 
        # <|ref|> Object Name <|/ref|> <|det|> [[x1, y1, x2, y2]] <|/det|>
        # BUT for OCR mode "Convert to markdown", it often outputs:
        # <|det|>[[x1,y1,x2,y2]]<|ref|>Text Content<|/ref|>
        
        # Flexible patterns for two common orderings:
        # 1) <|det|>[[x1,y1,x2,y2]]<|ref|>text<|/ref|>
        # 2) <|ref|>text<|/ref|><|det|>[[x1,y1,x2,y2]] (optionally followed by <|/det|>)
        patterns = [
            re.compile(
                r"<\|det\|>\s*\[\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\]\s*<\|ref\|>(.*?)<\|/ref\|>",
                re.DOTALL,
            ),
            re.compile(
                r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>\s*\[\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\](?:\s*<\|/det\|>)?",
                re.DOTALL,
            ),
        ]

        matches = []
        for pat in patterns:
            matches.extend(list(pat.finditer(text)))
        found_any = False
        
        for match in matches:
            found_any = True
            groups = match.groups()
            # pattern[0] = det first
            if len(groups) == 5 and groups[0].isdigit():
                x1, y1, x2, y2 = map(int, groups[:4])
                content = groups[4].strip()
            else:
                # pattern[1] = ref first
                content = groups[0].strip()
                x1, y1, x2, y2 = map(int, groups[1:5])
            
            if not content:
                continue

            max_coord = max(x1, y1, x2, y2)
            scale_x, scale_y, coord_mode = _coord_to_page_scale(max_coord)

            # Convert to page coordinates
            bbox_x = x1 * scale_x
            bbox_y = y1 * scale_y
            bbox_w = (x2 - x1) * scale_x
            bbox_h = (y2 - y1) * scale_y
            
            # Split by newlines to fix "huge bbox" issue
            lines = content.split("\n")
            lines = [line.strip() for line in lines if line.strip()]
            
            if not lines:
                continue
                
            if len(lines) == 1:
                # Single line, use full bbox
                blocks.append(TextBlock(
                    text=lines[0],
                    bbox={"x": bbox_x, "y": bbox_y, "width": bbox_w, "height": bbox_h},
                    metadata={
                        "ocr_engine": "deepseek",
                        "bbox_source": "grounded",
                        "grounding_coord_mode": coord_mode,
                    },
                ))
            else:
                # Multiple lines: split height evenly
                # This is a heuristic; ideally we'd have per-line grounding, but this is better than one huge box.
                line_height = bbox_h / len(lines)
                current_y = bbox_y
                
                for line in lines:
                    blocks.append(TextBlock(
                        text=line,
                        bbox={"x": bbox_x, "y": current_y, "width": bbox_w, "height": line_height},
                        metadata={
                            "ocr_engine": "deepseek",
                            "bbox_source": "grounded_split",
                            "grounding_coord_mode": coord_mode,
                        },
                    ))
                    current_y += line_height

        if not found_any:
            # Fallback if regex didn't match (maybe different format or no grounding tokens)
            logger.warning("[DeepSeek] Grounding enabled but no tokens found/parsed. Falling back to heuristic.")
            return self._create_text_blocks_from_ocr(text, image_size, target_size)

        logger.info("[DeepSeek] Parsed %d text blocks from grounded output", len(blocks))
        return blocks

    def _is_repetition_loop(self, text: str) -> bool:
        if not text or len(text) < 50:
            return False
        words = text.split()
        if len(words) < 10:
            return False
        word_counts = {}
        for word in words:
            word_clean = re.sub(r'[^\w]', '', word.lower())
            if word_clean:
                word_counts[word_clean] = word_counts.get(word_clean, 0) + 1
        if not word_counts:
            return False
        max_count = max(word_counts.values())
        if max_count / len(words) > 0.4:
            return True
        for i in range(len(words) - 4):
            window = words[i:i+5]
            window_clean = [re.sub(r'[^\w]', '', w.lower()) for w in window]
            if len(set(window_clean)) == 1 and window_clean[0]:
                return True
        return False
    
    def _fix_repetition_loops(self, text: str) -> str:
        if not text or len(text) < 20:
            return text
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            if not line.strip():
                cleaned_lines.append(line)
                continue
            words = line.split()
            if len(words) >= 3:
                cleaned_words = []
                prev_word = None
                repeat_count = 0
                for word in words:
                    word_normalized = re.sub(r'[^\w]', '', word.lower())
                    prev_normalized = re.sub(r'[^\w]', '', prev_word.lower()) if prev_word else None
                    if word_normalized == prev_normalized and word_normalized:
                        repeat_count += 1
                        if repeat_count >= 2:
                            continue
                    else:
                        repeat_count = 0
                    cleaned_words.append(word)
                    prev_word = word
                cleaned_line = ' '.join(cleaned_words)
            else:
                cleaned_line = line
            if cleaned_line:
                pattern = r'(\b\w+(?:\s+\w+)*\b)(?:\s*[,，。]\s*|\s+)(\1)(?:\s*[,，。]\s*|\s+)(\1)+'
                cleaned_line = re.sub(pattern, r'\1', cleaned_line)
            if cleaned_line.strip():
                if len(cleaned_line.strip()) < len(line.strip()) * 0.3 and len(line.strip()) > 50:
                    continue
                cleaned_lines.append(cleaned_line)
        cleaned_text = '\n'.join(cleaned_lines)
        if len(cleaned_text) > 100:
            sentence_endings = re.finditer(r'[.!?。！？]\s+', cleaned_text)
            endings = list(sentence_endings)
            if len(endings) > 2:
                last_portion_start = int(len(cleaned_text) * 0.7)
                last_portion = cleaned_text[last_portion_start:]
                last_words = last_portion.split()
                if len(last_words) > 10:
                    unique_words = len(set(w.lower() for w in last_words))
                    if unique_words / len(last_words) < 0.3:
                        if endings:
                            truncate_at = endings[-2].end() if len(endings) >= 2 else len(cleaned_text) * 0.7
                            cleaned_text = cleaned_text[:int(truncate_at)]
        return cleaned_text

    def _quality_metrics(self, text: str) -> Dict[str, float]:
        if not text:
            return {
                "char_count": 0.0,
                "nonprintable_ratio": 1.0,
                "alnum_ratio": 0.0,
                "url_like_ratio": 0.0,
            }

        chars = list(text)
        n = len(chars)
        nonprintable = sum(
            1
            for c in chars
            if (ord(c) < 9 or (ord(c) < 32 and c not in "\n\t\r"))
        )
        alnum = sum(1 for c in chars if c.isalnum())

        tokens = re.split(r"\s+", text.strip())
        url_like = sum(
            1
            for t in tokens
            if ("http" in t.lower() or "www." in t.lower() or ".com" in t.lower())
        )
        return {
            "char_count": float(n),
            "nonprintable_ratio": nonprintable / max(1, n),
            "alnum_ratio": alnum / max(1, n),
            "url_like_ratio": url_like / max(1, len(tokens)),
        }

    def _is_sane_output(self, text: str) -> Tuple[bool, str, Dict[str, float]]:
        from config.settings import settings

        metrics = self._quality_metrics(text)
        if metrics["char_count"] < float(getattr(settings, "deepseek_min_chars_quick", 500)):
            return False, "low_chars", metrics
        if metrics["nonprintable_ratio"] > float(
            getattr(settings, "deepseek_max_nonprintable_ratio", 0.02)
        ):
            return False, "too_many_nonprintable", metrics
        if metrics["alnum_ratio"] < float(
            getattr(settings, "deepseek_min_alnum_ratio", 0.12)
        ):
            return False, "low_alnum_ratio", metrics
        if metrics["url_like_ratio"] > float(
            getattr(settings, "deepseek_max_url_like_ratio", 0.01)
        ):
            return False, "looks_like_web_junk", metrics
        if self._is_repetition_loop(text):
            return False, "repetition_loop", metrics
        return True, "ok", metrics

    def _create_text_blocks_from_ocr(
        self,
        text: str,
        image_size: Tuple[int, int],
        target_size: Tuple[float, float],
    ) -> List[TextBlock]:
        """Fallback: line-based splitting for tighter diff highlights."""
        blocks: List[TextBlock] = []
        if not text or not text.strip():
            return blocks
        page_w, page_h = target_size

        # Prefer line-based blocks. If the model outputs markdown, this yields
        # much tighter boxes than paragraph-based heuristics.
        raw_lines = [ln.strip() for ln in text.split("\n")]
        lines = [ln for ln in raw_lines if ln]
        if not lines:
            return blocks

        y = 72.0
        left_margin = 36.0
        max_width = page_w * 0.7 if page_w > 0 else 500.0
        line_height = 14.0
        max_chars_per_line = 80
        char_width = 6.0

        for ln in lines:
            # If a line is very long, split into chunks to avoid huge boxes.
            chunks = [ln[i:i + max_chars_per_line] for i in range(0, len(ln), max_chars_per_line)]
            for chunk in chunks:
                n_chars = len(chunk)
                est_width = min(max_width, max(50.0, n_chars * char_width))
                block = TextBlock(
                    text=chunk,
                    bbox={"x": left_margin, "y": y, "width": est_width, "height": line_height},
                    style=None,
                    metadata={
                        "ocr_engine": "deepseek",
                        "bbox_source": "approx_line",
                        "bbox_units": "pt",
                        "bbox_space": "page",
                    },
                )
                blocks.append(block)
                y += line_height + 4.0

            # Small extra gap between logical lines.
            y += 2.0

            # Stop if we run off the page (avoid nonsense bboxes).
            if page_h and y > page_h - 36.0:
                break

        logger.info("[DeepSeek] Created %d text blocks from OCR output (heuristic)", len(blocks))
        return blocks


# =============================================================================
# SUBPROCESS WORKER FOR HARD TIMEOUT
# =============================================================================

def _subprocess_infer_worker(args: Dict) -> None:
    """
    Worker function that runs in a subprocess for hard timeout enforcement.
    
    This function loads the DeepSeek model fresh in the subprocess and runs inference.
    Results are written to a pickle file specified in args["result_path"].
    
    Args dict keys:
        - model_path: Path to DeepSeek model
        - image_path: Path to temporary image file
        - result_path: Path where result pickle will be written
        - prompt_template: Optional prompt template
        - base_size: Model base size
        - image_size: Model image size
        - use_grounding: Whether to use grounding tokens
    """
    import pickle
    import sys
    
    result = {"text": "", "peak_rss_mb": 0.0}
    
    try:
        import psutil

        # Lazy imports inside subprocess
        from pathlib import Path as SubPath
        
        model_path = args["model_path"]
        image_path = args["image_path"]
        prompt_template = args.get("prompt_template", "")
        base_size = args.get("base_size", 512)
        image_size = args.get("image_size", 512)
        use_grounding = args.get("use_grounding", False)
        
        # Load model in subprocess
        ocr_engine = DeepSeekOCR(model_path)
        ocr_engine._load_model()
        
        if ocr_engine._model is None or ocr_engine._tokenizer is None:
            result["error"] = "model_not_loaded"
            with open(args["result_path"], "wb") as f:
                pickle.dump(result, f)
            return
        
        # Build prompt
        prompt = "<image>\n"
        if use_grounding:
            prompt += "<|grounding|>"
        if prompt_template:
            prompt += prompt_template
        else:
            prompt += "Perform OCR and return the result in reading order."
        
        # Create output directory
        output_path = SubPath(tempfile.gettempdir()) / "deepseek_subprocess_out"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Run inference
        infer_result = ocr_engine._model.infer(
            tokenizer=ocr_engine._tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=str(output_path),
            base_size=base_size,
            image_size=image_size,
            crop_mode=False,
            save_results=False,
            test_compress=False,
            eval_mode=True,
        )
        
        result["text"] = infer_result[0] if infer_result else ""
        result["peak_rss_mb"] = psutil.Process().memory_info().rss / (1024 * 1024)
        
    except Exception as e:
        result["error"] = str(e)[:200]
    
    # Write result
    try:
        with open(args["result_path"], "wb") as f:
            pickle.dump(result, f)
    except Exception:
        sys.exit(1)


_ocr_instance: Optional[DeepSeekOCR] = None

def get_ocr_instance(model_path: str) -> DeepSeekOCR:
    global _ocr_instance
    if _ocr_instance is None or _ocr_instance.model_path != model_path:
        _ocr_instance = DeepSeekOCR(model_path)
    return _ocr_instance

def ocr_pdf(path: str | Path, model_path: str) -> List[PageData]:
    path = Path(path)
    logger.info("[DeepSeek] Running OCR on PDF: %s", path)
    import time
    from config.settings import settings
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for OCR rendering.") from exc
    doc = fitz.open(path)
    ocr = get_ocr_instance(model_path)
    pages: List[PageData] = []
    for page in doc:
        page_num = page.number + 1
        render_start = time.time()
        dpi = int(getattr(settings, "deepseek_render_dpi", 60))
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        render_time = time.time() - render_start
        ocr_start = time.time()
        text_blocks = ocr.recognize(pix, target_size=(page.rect.width, page.rect.height))
        ocr_time = time.time() - ocr_start

        if (
            getattr(settings, "adaptive_ocr_enabled", True)
            and (not text_blocks)
            and getattr(ocr, "_last_reject_reason", None)
        ):
            dpi2 = int(getattr(settings, "deepseek_retry_dpi", 96))
            if dpi2 > dpi:
                logger.info(
                    "[DeepSeek] Retrying page %d with higher DPI=%d (reason=%s)",
                    page_num,
                    dpi2,
                    getattr(ocr, "_last_reject_reason", None),
                )
                render_start = time.time()
                pix2 = page.get_pixmap(dpi=dpi2, alpha=False)
                render_time += time.time() - render_start
                ocr_start = time.time()
                text_blocks = ocr.recognize(
                    pix2, target_size=(page.rect.width, page.rect.height)
                )
                ocr_time += time.time() - ocr_start
                dpi = dpi2
        logger.info("[DeepSeek] Page %d processed in %.2fs (%d blocks)", page_num, ocr_time, len(text_blocks))
        page_data = PageData(
            page_num=page_num,
            width=page.rect.width,
            height=page.rect.height,
            blocks=text_blocks,
        )
        page_data.metadata = {
            "extraction_method": "ocr_deepseek",
            "ocr_engine_used": "deepseek",
            "dpi": dpi,
            "render_time": render_time,
            "ocr_time": ocr_time,
            "deepseek_reject_reason": getattr(ocr, "_last_reject_reason", None),
            "deepseek_quality": getattr(ocr, "_last_quality", {}),
        }
        if not text_blocks:
            page_data.metadata["ocr_fallback_reason"] = "empty_result"
        pages.append(page_data)
    doc.close()
    logger.info("[DeepSeek] OCR processed %d pages", len(pages))
    return pages
