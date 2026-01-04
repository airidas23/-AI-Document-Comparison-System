# DeepSeek-OCR MPS (Apple Silicon) Fix

## Problema

DeepSeek-OCR modelis ant Apple Silicon (M1–M4) gali klausti dėl dtype mismatch klaidų:
- `masked_scatter expected ... got BFloat16 and Float`
- Dtype konfliktai tarp vaizdo embedding'ų ir tekstinių embedding'ų

## Sprendimas: "Nuclear Reset"

### 1. Išvalyti seną modelį ir cache

```bash
# Ištrinti lokalią kopiją
rm -rf "models/deepseek-ocr"

# Ištrinti HuggingFace cache
rm -rf ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR
rm -rf ~/.cache/huggingface/modules/transformers_modules/deepseek-ai/DeepSeek-OCR
```

### 2. Parsisiųsti naują modelį

Naudokite automatinį skriptą:

```bash
python scripts/download_deepseek_ocr_fresh.py --revision 1e3401a3d4603e9e71ea0ec850bfead602191ec4
```

Arba rankiniu būdu:

```bash
# Su huggingface-cli (jei įdiegtas)
huggingface-cli download deepseek-ai/DeepSeek-OCR \
  --local-dir "models/deepseek-ocr" \
  --local-dir-use-symlinks False

# Arba su Python
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='deepseek-ai/DeepSeek-OCR', local_dir='models/deepseek-ocr', local_dir_use_symlinks=False)"
```

### 3. Patikrinti modelį

```bash
python scripts/verify_deepseek_model.py
```

Turi būti:
- ✓ Model type: `deepseek_vl_v2` su `DeepseekOCR` architektūra
- ✓ dtype/device conversion kodas modeling_deepseekocr.py (eilutės ~567-570)
- ✓ Modelio svoriai (.safetensors) parsisiųsti

### 4. Patikrinti ar MPS veikia

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

## Techninė informacija

### Kodėl tai veikia?

1. **Naujausias modelis turi MPS palaikymą**: DeepSeek pridėjo commit'ą "Add Apple Silicon (MPS) backend support" (2025-01-21)

2. **Dtype suvienodinimas**: `modeling_deepseekocr.py` eilutėse 567-570 yra kodas, kuris priverstinai suvienodina dtype:
   ```python
   target = inputs_embeds[idx]
   images_in_this_batch = images_in_this_batch.to(
       device=target.device,
       dtype=target.dtype,
   )
   ```

3. **MPS dtype pasirinkimas**: Ant MPS naudojame `float32` (ne `bfloat16`), nes:
   - MPS istorijoje bf16 palaikymas buvo nestabilus
   - float32 yra saugesnis pasirinkimas ant MPS
   - Naujausias DeepSeek-OCR kodas tai tvarko automatiškai

### Pataisymai `deepseek_ocr_engine.py`

1. **Sub-modelių dtype suvienodinimas**: Užtikriname, kad SAM, vision_model ir projector būtų to paties dtype kaip pagrindinis modelis:
   ```python
   if hasattr(self._model, 'sam_model') and self._model.sam_model is not None:
       self._model.sam_model = self._model.sam_model.to(device).to(dtype)
   if hasattr(self._model, 'vision_model') and self._model.vision_model is not None:
       self._model.vision_model = self._model.vision_model.to(device).to(dtype)
   if hasattr(self._model, 'projector') and self._model.projector is not None:
       self._model.projector = self._model.projector.to(device).to(dtype)
   ```

2. **MPS dtype pasirinkimas**: Ant MPS naudojame `float32`:
   ```python
   elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
       device = torch.device("mps")
       dtype_candidates = [torch.float32]  # Saugiau nei bfloat16 ant MPS
   ```

## Testavimas

Po reset'o, patikrinkite ar modelis veikia:

```python
from extraction.deepseek_ocr_engine import DeepSeekOCR
import torch

# Patikrinti MPS
print("MPS available:", torch.backends.mps.is_available())

# Įkelti modelį
ocr = DeepSeekOCR("models/deepseek-ocr")
print("Model loaded successfully!")

# Testuoti su vaizdu (jei turite)
# from PIL import Image
# img = Image.open("test_image.jpg")
# blocks = ocr.recognize(img)
# print(f"Found {len(blocks)} text blocks")
```

## Jei vis tiek neveikia

1. **Patikrinkite error stacktrace**: Jei vis dar gaunate dtype klaidas, įkelkite pilną stacktrace (nuo viršaus iki apačios)

2. **Patikrinkite PyTorch versiją**: Rekomenduojama `torch>=2.0.0` su MPS palaikymu

3. **Patikrinkite transformers versiją**: Rekomenduojama `transformers>=4.46.3`

4. **Priverstinai naudokite CPU** (lėtai, bet turėtų veikti):
   ```python
   # Temporarily force CPU for testing
   import os
   os.environ["CUDA_VISIBLE_DEVICES"] = ""
   # Then test...
   ```

## Nuorodos

- [DeepSeek-OCR HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [DeepSeek-OCR-Cli (community tool)](https://github.com/opendatalab/DeepSeekOCR-Cli) - veikia "natively on Apple Silicon (M1-M4) using PyTorch MPS"
- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)

