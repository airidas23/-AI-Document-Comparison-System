#!/usr/bin/env python3
"""Download DeepSeek-OCR model fresh from HuggingFace with cache clearing.

Notes:
- Uses a project-relative `models/deepseek-ocr` directory by default.
- Supports pinning a Hugging Face `revision` for deterministic installs.
"""

import argparse
import shutil
from pathlib import Path


DEFAULT_REPO_ID = "deepseek-ai/DeepSeek-OCR"
DEFAULT_REVISION = "1e3401a3d4603e9e71ea0ec850bfead602191ec4"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download DeepSeek-OCR model fresh with cache clearing")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Hugging Face git revision/commit hash (pin for deterministic installs)",
    )
    parser.add_argument(
        "--model-dir",
        default=str(_project_root() / "models" / "deepseek-ocr"),
        help="Local directory for model files",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    
    # Step 1: Remove local model if exists
    if model_dir.exists():
        print(f"Removing existing model at {model_dir}")
        shutil.rmtree(model_dir)
    
    # Step 2: Clear HF cache
    cache_paths = [
        Path.home() / ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR",
        Path.home() / ".cache/huggingface/modules/transformers_modules/deepseek-ai/DeepSeek-OCR",
    ]
    
    for cache_path in cache_paths:
        if cache_path.exists():
            print(f"Removing cache at {cache_path}")
            shutil.rmtree(cache_path)
    
    # Step 3: Download fresh model
    print("Downloading DeepSeek-OCR from HuggingFace...")
    try:
        from huggingface_hub import snapshot_download
        
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=str(args.repo_id),
            revision=str(args.revision) if args.revision else None,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
        print(f"✓ Model downloaded to {model_dir}")
    except ImportError:
        print("ERROR: huggingface_hub not installed. Install with: pip install huggingface_hub")
        return 1
    except Exception as e:
        print(f"ERROR: Failed to download model: {e}")
        return 1
    
    # Step 4: Verify model type
    print("\nVerifying model type...")
    try:
        from transformers import AutoConfig
        
        config = AutoConfig.from_pretrained(
            str(model_dir),
            trust_remote_code=True
        )
        
        model_type = getattr(config, 'model_type', None)
        architectures = getattr(config, 'architectures', [])
        auto_map = getattr(config, 'auto_map', {})
        
        print(f"  Model type: {model_type}")
        print(f"  Architectures: {architectures}")
        
        has_ocr_arch = any('DeepseekOCR' in str(arch) for arch in architectures)
        has_ocr_auto = any('DeepseekOCR' in str(v) for v in auto_map.values())
        
        if model_type == "deepseek_vl_v2" and (has_ocr_arch or has_ocr_auto):
            print("✓ Model is correct: DeepSeek-OCR (uses deepseek_vl_v2 base)")
            return 0
        elif has_ocr_arch or has_ocr_auto:
            print("✓ Model is correct: DeepSeek-OCR detected")
            return 0
        else:
            print("⚠ WARNING: Model type verification unclear")
            print(f"  Expected: DeepseekOCR architecture")
            print(f"  Got: model_type={model_type}, architectures={architectures}")
            return 1
    except Exception as e:
        print(f"ERROR: Failed to verify model: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

