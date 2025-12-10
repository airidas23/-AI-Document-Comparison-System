
import os
from pathlib import Path

target_file = Path("models/deepseek-ocr/modeling_deepseekocr.py")

if not target_file.exists():
    print(f"Error: {target_file} not found")
    exit(1)

content = target_file.read_text(encoding="utf-8")

new_lines = []
lines = content.split('\n')
for line in lines:
    # 1. Fix temperature warning: Remove temperature=0.0, ensure do_sample=False
    if "temperature=0.0," in line:
        line = line.replace("temperature=0.0,", "do_sample=False,")
    
    # 2. Fix pad_token_id warning: It uses eos_token_id already?
    # The code has: eos_token_id=tokenizer.eos_token_id,
    # The warning says: Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    # We should add pad_token_id=tokenizer.eos_token_id explicity if missing.
    if "eos_token_id=tokenizer.eos_token_id," in line:
        if "pad_token_id" not in line:
            line = line + " pad_token_id=tokenizer.eos_token_id,"

    # 3. Fix attention_mask warning:
    # The warning says: The attention mask and the pad token id were not set.
    # We construct input_ids internally. We should also construct attention_mask.
    # In `prepare_inputs_for_generation`, it might expect it.
    # But `generate` is called with `input_ids`.
    # Code:
    # input_ids = torch.LongTensor(tokenized_str)
    # output_ids = self.generate(input_ids.unsqueeze(0).to(self.device), ...)
    #
    # We can create a simple attention mask of ones since we don't seem to have padding in input_ids here (it's a single sequence).
    # input_ids shape is [1, seq_len] after unsqueeze.
    # mask should be ones of same shape.
    
    # Finding the generate call to inject attention_mask
    if "images_seq_mask = images_seq_mask.unsqueeze(0).to(self.device)," in line:
         # Check if we already added it (idempotency)
         if "attention_mask" not in line:
             # We need to construct it first.
             # Wait, editing python code line by line is fragile for variable creation.
             # Better to add `attention_mask` argument to `generate` call.
             # The `input_ids` on previous line is `input_ids.unsqueeze(0).to(self.device)`
             # `generate` signature allows `attention_mask=...`
             # We can pass `attention_mask=input_ids.ne(tokenizer.pad_token_id or -100).unsqueeze(0).to(self.device)`?
             # Or just `attention_mask=torch.ones_like(input_ids.unsqueeze(0)).to(self.device)`
             # But `input_ids` variable used in `generate` is `input_ids.unsqueeze(0).to(self.device)`
             # Accessing it inside `generate` call arguments is hard if it's an expression.
             
             # Let's look at the `generate` call:
             # output_ids = self.generate(
             #    input_ids.unsqueeze(0).to(self.device),
             #    ...
             # )
             #
             # We can add `attention_mask=torch.ones((1, input_ids.shape[0]), device=self.device, dtype=torch.long),`
             pass
             
    new_lines.append(line)

# Advanced replacement for attention mask
# locating the generate block
processed_content = '\n'.join(new_lines)

# We need to insert attention_mask creation before generate, or pass it in generate.
# Let's simple pass `attention_mask=torch.ones_like(input_ids.unsqueeze(0)).to(self.device)` logic?
# But `torch` might not be imported or `input_ids` not available in scope? `input_ids` IS available.
#
# Let's use string replace for the generate call to add attention_mask.
# Pattern: input_ids.unsqueeze(0).to(self.device),
# Replace with: input_ids.unsqueeze(0).to(self.device), attention_mask=torch.ones_like(input_ids.unsqueeze(0)).to(self.device),

processed_content = processed_content.replace(
    "input_ids.unsqueeze(0).to(self.device),",
    "input_ids.unsqueeze(0).to(self.device), attention_mask=torch.ones_like(input_ids.unsqueeze(0)).to(self.device),"
)

# Write back
target_file.write_text(processed_content, encoding="utf-8")
print(f"Successfully patched warnings in {target_file}")
