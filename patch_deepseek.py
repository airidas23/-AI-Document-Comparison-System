
import os
from pathlib import Path

target_file = Path("models/deepseek-ocr/modeling_deepseekocr.py")

if not target_file.exists():
    print(f"Error: {target_file} not found")
    exit(1)

content = target_file.read_text(encoding="utf-8")

# Helper to find indentation
def get_indent(line):
    return len(line) - len(line.lstrip())

new_lines = []
lines = content.split('\n')
for line in lines:
    # Replace .cuda() with .to(self.device)
    if ".cuda()" in line:
        line = line.replace(".cuda()", ".to(self.device)")
    
    # Replace torch.autocast("cuda"...)
    # We need to handle this carefully. cpu autocast is different.
    # We'll replace it with a dynamic check or just use self.device.type if possible, 
    # but autocast string must be literal usually or variable.
    # Let's verify what self.device.type returns. 'cpu', 'cuda', 'mps'.
    # torch.autocast(device_type=...) works with these.
    # However, 'mps' autocast was unstable in some versions.
    # Let's replace the hardcoded block with a dynamic context manager logic?
    # Or just replace the string "cuda" with self.device.type?
    # But wait, self.device is not available inside the `with` statement context easily unless it's a method.
    # `infer` is a method of `DeepseekOCR` (which subclasses PreTrainedModel/nn.Module), so `self.device` is available.
    if 'torch.autocast("cuda"' in line:
        # Replacement: use device type, but fallback to 'cpu' if issue?
        # Simpler: just change the string "cuda" to self.device.type
        # But we need to handle the quote.
        # "cuda" -> self.device.type
        line = line.replace('"cuda"', 'self.device.type')
    
    new_lines.append(line)

new_content = '\n'.join(new_lines)

# Write back
target_file.write_text(new_content, encoding="utf-8")
print(f"Successfully patched {target_file}")
