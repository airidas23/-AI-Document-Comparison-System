"""Persistent DeepSeek OCR worker optimized for Apple Silicon M4.

This module implements a long-lived subprocess that keeps the DeepSeek model loaded
on MPS (Metal Performance Shaders), eliminating model reload time per page.

M4-Specific Optimizations:
- Forced MPS backend (no CUDA fallback)
- float16 dtype (more stable than bfloat16 on MPS PyTorch backend)
- torch.mps.empty_cache() after each generation (critical for Unified Memory)
- torch.no_grad() to prevent gradient accumulation (saves RAM)

Usage:
    from extraction.deepseek_persistent_worker import get_persistent_worker
    
    worker = get_persistent_worker(model_path)
    raw_text, elapsed, peak_rss = worker.infer(img, prompt, ...)
"""
import multiprocessing
import time
import traceback
import sys
import os
import psutil
from typing import Optional, Tuple
from PIL import Image
import numpy as np

from utils.logging import logger


# --- SINGLETON MANAGEMENT ---
_GLOBAL_WORKER = None


def get_persistent_worker(model_path: str) -> "DeepSeekPersistentWorker":
    """Get or create a singleton persistent worker for the given model path."""
    global _GLOBAL_WORKER
    if _GLOBAL_WORKER is None:
        _GLOBAL_WORKER = DeepSeekPersistentWorker(model_path)
    return _GLOBAL_WORKER


def shutdown_all_workers():
    """Shutdown the global worker."""
    global _GLOBAL_WORKER
    if _GLOBAL_WORKER is not None:
        _GLOBAL_WORKER.stop()
        _GLOBAL_WORKER = None


class DeepSeekPersistentWorker:
    """Manager for persistent DeepSeek worker subprocess optimized for M4."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.process: Optional[multiprocessing.Process] = None
        self.input_queue: Optional[multiprocessing.Queue] = None
        self.output_queue: Optional[multiprocessing.Queue] = None
        self.is_warm = False

    def _start_worker(self):
        if self.process is not None and self.process.is_alive():
            return

        print(f"[DeepSeekWorker M4] Starting dedicated MPS process...")
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()
        self.is_warm = False
        
        # Important: 'spawn' start method is required for PyTorch on Mac
        ctx = multiprocessing.get_context('spawn')
        self.process = ctx.Process(
            target=_worker_loop_m4,
            args=(self.input_queue, self.output_queue, self.model_path),
            daemon=True
        )
        self.process.start()
        logger.info("[DeepSeekWorker M4] Started worker subprocess (PID %d)", self.process.pid)

    def stop(self):
        """Gracefully stop the worker."""
        if self.process and self.process.is_alive():
            self.input_queue.put(("STOP", None))
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
        self.process = None
        self.is_warm = False

    def kill_and_restart(self):
        """Force kill and restart the worker."""
        print("[DeepSeekWorker M4] Force killing worker...")
        if self.process:
            self.process.kill()
            self.process.join()
        self.process = None
        self._start_worker()

    def infer(
        self,
        image_path: str,
        prompt_template: str,
        base_size: int,
        image_size: int,
        use_grounding: bool,
        timeout_sec: int,
        memory_hard_mb: int = 6000,
    ) -> Tuple[str, float, float]:
        """Run inference on an image.
        
        Args:
            image_path: Path to the image file
            prompt_template: Prompt for the model
            base_size: Base size for inference (unused in M4 path, kept for API compat)
            image_size: Image size for inference (unused in M4 path, kept for API compat)
            use_grounding: Whether to enable grounding mode
            timeout_sec: Timeout in seconds (extended for cold start)
            memory_hard_mb: Memory limit (unused, kept for API compat)
        
        Returns:
            Tuple of (raw_text, elapsed_seconds, peak_rss_mb)
        
        Raises:
            TimeoutError: If inference exceeds timeout
            RuntimeError: If worker fails
        """
        self._start_worker()

        # M4: First load (cold start) takes longer
        effective_timeout = timeout_sec + 60 if not self.is_warm else timeout_sec

        # Load image from path
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {e}")

        # Build prompt with grounding if needed
        full_prompt = prompt_template or "Perform OCR and return the result in reading order."
        if use_grounding:
            full_prompt = "<|grounding|>" + full_prompt

        # Send data
        payload = {
            "img": img,  # PIL Image
            "prompt_template": full_prompt,
        }

        try:
            self.input_queue.put(("INFER", payload))
            
            # Wait for response
            result = self.output_queue.get(timeout=effective_timeout)
            status, data = result
            
            if status == "SUCCESS":
                self.is_warm = True
                return data["raw_text"], data["elapsed"], data["peak_rss"]
            else:
                raise RuntimeError(f"DeepSeek M4 Error: {data}")

        except Exception as e:
            if "Empty" in type(e).__name__ or "timeout" in str(e).lower():
                print(f"[DeepSeekWorker M4] Timeout ({effective_timeout}s). Resetting MPS context.")
                self.kill_and_restart()
                raise TimeoutError("DeepSeek M4 worker timed out")
            raise

    def shutdown(self):
        """Alias for stop() for API compatibility."""
        self.stop()


# --- M4 SPECIFIC WORKER LOOP ---

def _worker_loop_m4(input_q, output_q, model_path):
    """
    Worker loop optimized for Apple Silicon M4.
    
    Uses:
    - MPS backend exclusively
    - float16 dtype (most stable on MPS)
    - torch.mps.empty_cache() after each generation
    - torch.no_grad() for memory efficiency
    """
    # 1. Environment variables for PyTorch MPS
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    import torch
    from janus.models import MultiModalityCausalLM, VLChatProcessor

    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore")

    print(f"[Worker M4 PID {os.getpid()}] Initializing MPS context...")

    # Model state
    model = None
    vl_chat_processor = None
    tokenizer = None
    
    # M4 Settings
    device = "mps"  # Force MPS
    # M4 supports bfloat16, but float16 is more stable in PyTorch MPS backend
    dtype = torch.float16 

    while True:
        try:
            cmd, payload = input_q.get()
        except EOFError:
            break

        if cmd == "STOP":
            break

        if cmd == "INFER":
            try:
                t0 = time.time()
                
                # --- A. MODEL LOADING (FIRST TIME ONLY) ---
                if model is None:
                    print(f"[Worker M4] Loading model to Unified Memory ({dtype})...")
                    
                    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
                    tokenizer = vl_chat_processor.tokenizer

                    # Load directly to MPS
                    model = MultiModalityCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True
                    )
                    model = model.to(dtype).to(device).eval()
                    
                    print(f"[Worker M4] Model Ready on {device}.")

                # --- B. IMAGE PROCESSING ---
                img_input = payload['img']
                
                # Convert Numpy -> PIL (Required for DeepSeek)
                if isinstance(img_input, np.ndarray):
                    # Normalize if needed
                    if img_input.dtype == np.float32 or img_input.dtype == np.float64:
                        img_input = (img_input * 255).astype(np.uint8)
                    pil_image = Image.fromarray(img_input).convert("RGB")
                elif hasattr(img_input, 'convert'):
                    pil_image = img_input.convert("RGB")
                else:
                    pil_image = img_input

                # --- C. GENERATION LOGIC (MPS SAFE) ---
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": f"<image_placeholder>\n{payload['prompt_template']}",
                        "images": [pil_image],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]

                # Prepare inputs
                prepare_inputs = vl_chat_processor(
                    conversations=conversation, 
                    images=[pil_image], 
                    force_batchify=True
                ).to(device, dtype=dtype)  # Important: send to MPS

                inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

                # Generate with no_grad (saves memory)
                with torch.no_grad():
                    outputs = model.language_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=prepare_inputs.attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=4000,
                        do_sample=False,  # Greedy decoding - fastest and most stable for OCR
                        use_cache=True,
                    )

                # Decode
                answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                
                # --- D. CLEANUP ---
                # Critical step for M4: clear MPS cache to prevent memory clogging
                torch.mps.empty_cache()

                elapsed = time.time() - t0
                mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

                # Debug
                if not answer:
                    print("[Worker M4 WARNING] Generated empty text!")
                else:
                    print(f"[Worker M4 SUCCESS] Generated {len(answer)} chars in {elapsed:.1f}s")
                
                output_q.put(("SUCCESS", {
                    "raw_text": answer,
                    "elapsed": elapsed,
                    "peak_rss": mem_usage
                }))

            except Exception as e:
                print(f"[Worker M4 ERROR] {e}")
                traceback.print_exc()
                output_q.put(("ERROR", str(e)))
                # Important: if MPS error occurred, often need to clear cache
                try:
                    import torch
                    if "mps" in str(e).lower():
                        torch.mps.empty_cache()
                except Exception:
                    pass


# Register cleanup on interpreter exit
import atexit
atexit.register(shutdown_all_workers)
