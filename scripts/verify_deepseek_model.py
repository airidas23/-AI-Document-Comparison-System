#!/usr/bin/env python3
"""Verify that DeepSeek-OCR model is correct and has MPS support."""
import sys
from pathlib import Path


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "deepseek-ocr"

def main():
    model_dir = DEFAULT_MODEL_DIR
    
    if not model_dir.exists():
        print(f"ERROR: Model directory does not exist: {model_dir}")
        print("Run: python scripts/download_deepseek_ocr_fresh.py first")
        return 1
    
    print(f"Checking model at: {model_dir}\n")
    
    # Check 1: Model files exist
    required_files = ["config.json", "modeling_deepseekocr.py", "modeling_deepseekv2.py"]
    missing = [f for f in required_files if not (model_dir / f).exists()]
    if missing:
        print(f"ERROR: Missing required files: {missing}")
        return 1
    print("✓ All required files present")
    
    # Check 2: Model type
    try:
        from transformers import AutoConfig
        
        config = AutoConfig.from_pretrained(
            str(model_dir),
            trust_remote_code=True
        )
        
        model_type = getattr(config, 'model_type', None)
        architectures = getattr(config, 'architectures', [])
        auto_map = getattr(config, 'auto_map', {})
        
        print(f"\nModel configuration:")
        print(f"  Model type: {model_type}")
        print(f"  Architectures: {architectures}")
        
        has_ocr_arch = any('DeepseekOCR' in str(arch) for arch in architectures)
        has_ocr_auto = any('DeepseekOCR' in str(v) for v in auto_map.values())
        
        if model_type == "deepseek_vl_v2" and (has_ocr_arch or has_ocr_auto):
            print("✓ Model is correct: DeepSeek-OCR (uses deepseek_vl_v2 base)")
        elif has_ocr_arch or has_ocr_auto:
            print("✓ Model is correct: DeepSeek-OCR detected")
        else:
            print("⚠ WARNING: Model type verification unclear")
            print(f"  Expected: DeepseekOCR architecture")
            return 1
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        return 1
    
    # Check 3: MPS support in modeling_deepseekocr.py
    modeling_file = model_dir / "modeling_deepseekocr.py"
    if modeling_file.exists():
        content = modeling_file.read_text(encoding='utf-8', errors='ignore')
        
        # Heuristics for Apple Silicon friendliness.
        # NOTE: Hardcoded `.cuda()` typically breaks on Mac (no CUDA).
        has_masked_scatter = "masked_scatter" in content
        has_cuda_hardcode = ".cuda()" in content
        has_device_aware_to = "to(" in content and ("target.device" in content or "self.device" in content)
        
        print(f"\nMPS support checks:")
        if has_masked_scatter:
            print("✓ masked_scatter found (image embedding insertion)")
            if has_cuda_hardcode:
                print("⚠ WARNING: Found hardcoded .cuda() calls (likely to break on Apple Silicon)")
            else:
                print("✓ No hardcoded .cuda() calls found")
        else:
            print("⚠ WARNING: masked_scatter not found")
            return 1
        
        if has_device_aware_to:
            print("✓ Device-aware tensor moves detected")
        else:
            print("⚠ WARNING: No obvious device-aware tensor moves detected")
    else:
        print("⚠ WARNING: modeling_deepseekocr.py not found (remote-code model?)")
    
    # Check 4: Weights exist
    safetensors = list(model_dir.glob("*.safetensors"))
    if safetensors:
        total_size = sum(f.stat().st_size for f in safetensors) / (1024**3)
        print(f"\n✓ Model weights found: {len(safetensors)} file(s), {total_size:.2f} GB")
    else:
        print("\n⚠ WARNING: No .safetensors files found")
        return 1
    
    print("\n" + "="*60)
    print("✓ Model verification complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Test the model with: python -c \"from extraction.deepseek_ocr_engine import DeepSeekOCR; ocr = DeepSeekOCR('models/deepseek-ocr'); print('Model loaded successfully')\"")
    print("2. If you get dtype errors, ensure you're using the latest model from HuggingFace")
    print("3. Check that PyTorch MPS is available: python -c \"import torch; print('MPS available:', torch.backends.mps.is_available())\"")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

