# DeepSeek-OCR Model Fix

## Problem

The local `models/deepseek-ocr/` folder contained a **DeepSeek-VL2** checkpoint instead of **DeepSeek-OCR**, causing confusion.

**Important Note**: DeepSeek-OCR actually uses `deepseek_vl_v2` as its base architecture (this is normal!), but it should have:
- `architectures: ["DeepseekOCRForCausalLM"]` - OCR-specific architecture class
- `auto_map` pointing to `modeling_deepseekocr.DeepseekOCRForCausalLM` - OCR wrapper

The issue was that the folder had a pure DeepSeek-VL2 model (without the OCR wrapper), not the DeepSeek-OCR model (which is VL2 + OCR wrapper).

## Solution Applied

### 1. Renamed Incorrect Folder
- Moved `models/deepseek-ocr/` → `models/deepseek-vl2-accident/`
- This prevents the code from accidentally loading the wrong model

### 2. Added Model Type Validation

#### In `scripts/setup_models.py`:
- Added `validate_model_type()` function to check model type after download
- Updated `download_model()` to accept `expected_model_type` parameter
- Validates existing models before skipping download

#### In `extraction/deepseek_ocr_engine.py`:
- Added validation before loading local models
- Checks if `model_type == 'deepseekocr'` (case-insensitive)
- Falls back to HuggingFace hub if wrong model type detected
- Logs clear error messages explaining the issue

### 3. Created Download Script
- `scripts/download_deepseek_ocr.py` - Quick script to download and validate the correct model

## How to Download the Correct Model

### Option 1: Using the Download Script (Recommended)

```bash
cd "/Users/airidas/Documents/KTU/P170M109 Computational Intelligence and Decision Making/project"
python3 scripts/download_deepseek_ocr.py
```

### Option 2: Using the Setup Script

```bash
cd "/Users/airidas/Documents/KTU/P170M109 Computational Intelligence and Decision Making/project"
python3 scripts/setup_models.py
```

### Option 3: Using HuggingFace CLI

```bash
cd "/Users/airidas/Documents/KTU/P170M109 Computational Intelligence and Decision Making/project"
huggingface-cli download deepseek-ai/DeepSeek-OCR \
  --local-dir models/deepseek-ocr \
  --local-dir-use-symlinks False
```

### Option 4: Using Python

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="deepseek-ai/deepseek-ocr",
    local_dir="models/deepseek-ocr",
    local_dir_use_symlinks=False,
)
```

## Verification

After downloading, verify the model:

```bash
python3 -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('models/deepseek-ocr', trust_remote_code=True); print(f'Model type: {c.model_type}'); print(f'Architectures: {c.architectures}')"
```

**Expected output**:
- `Model type: deepseek_vl_v2` (this is CORRECT - DeepSeek-OCR uses VL2 as base architecture)
- `Architectures: ['DeepseekOCRForCausalLM']` (this confirms it's the OCR model)

**Note**: The warning "You are using a model of type deepseek_vl_v2 to instantiate a model of type DeepseekOCR" is normal and can be ignored. DeepSeek-OCR uses `deepseek_vl_v2` as its base architecture but has the `DeepseekOCRForCausalLM` architecture class for OCR-specific functionality.

## Model Repositories

- **DeepSeek-OCR**: `deepseek-ai/DeepSeek-OCR` (correct)
- **DeepSeek-VL2**: `deepseek-ai/deepseek-vl2` (wrong - this was accidentally downloaded)

## Prevention

The code now:
1. Validates model type before loading local models
2. Falls back to HuggingFace hub if wrong model type detected
3. Provides clear error messages explaining the issue
4. Validates model type after download in setup scripts

## Status

- ✅ Incorrect folder renamed to `deepseek-vl2-accident`
- ✅ Model type validation added to engine
- ✅ Model type validation added to setup script
- ✅ Download script created
- ⏳ **Next step**: Download the correct model using one of the methods above

