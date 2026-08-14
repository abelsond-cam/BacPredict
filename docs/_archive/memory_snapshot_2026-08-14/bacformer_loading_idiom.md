---
name: bacformer-loading-idiom
description: "Canonical idiom for loading Bacformer models from Hugging Face — uses dtype=\"auto\" rather than manual bf16 cast"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 965bdc91-22aa-4d38-9677-c15707d973a6
---

When loading a Bacformer model from Hugging Face for inference or fine-tuning, use the canonical idiom:

```python
from transformers import AutoModelForMaskedLM  # or AutoModelForSequenceClassification for fine-tuning head
model = AutoModelForMaskedLM.from_pretrained(
    "macwiatrak/bacformer-large-masked-complete-genomes",
    trust_remote_code=True,
    dtype="auto",
)
```

The key bit is `dtype="auto"` — let HF pick precision from the model's HF config. **Do not** force-cast with `.to(torch.bfloat16)` (as the older [src/kleb_ast/train_amr.py](src/kleb_ast/train_amr.py) does) — that bf16 cast is slow on CPU and breaks the smoke-test / Stage A protocol that mandates CPU-disabled execution.

**Why:** User flagged this as the right idiom when reviewing the TB AST scaffolding plan. Manual bf16 cast was a workaround in the original Kleb code that doesn't survive the §0.2 three-stage testing protocol's Stage A CPU requirement.

**How to apply:** Any time we write or update a Bacformer fine-tuning entrypoint in this repo (`train_amr.py` or analogue under another task package), use `dtype="auto"` and skip the manual cast. When porting kleb_ast scaffolding to a new task (tb_ast, kleb_iso_source, etc.), upgrade this line — don't copy the old cast forward.

Related: [[refreshed-bacformer-complete-genomes-model]] for the model ID convention.
