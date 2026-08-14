---
name: refreshed-bacformer-complete-genomes-model
description: The Hugging Face model ID for the refreshed Bacformer complete-genomes weights used as the base for all BacPredict fine-tuning
metadata: 
  node_type: memory
  type: project
  originSessionId: 965bdc91-22aa-4d38-9677-c15707d973a6
---

All BacPredict experiments (Tasks 1–6) fine-tune from the **complete-genomes** Bacformer model — **not** the older MAG-trained one currently hardcoded in some scripts.

The refreshed Hugging Face model ID is:

```
macwiatrak/bacformer-large-masked-complete-genomes
```

**Why:** Root [CLAUDE.md](/Users/davidabelson/developer/BacPredict/CLAUDE.md) §0.1 mandates this — earlier local Bacformer weights had defects since fixed by the authors, and the MAG-trained variant is for a different use case. The user confirmed the exact HF ID when planning the TB AST work on 2026-05-25.

**How to apply:** Whenever scaffolding a new training entrypoint or updating default model IDs in this repo, default to `macwiatrak/bacformer-large-masked-complete-genomes`. Places to check / keep in sync:

- [src/tl/embed/generate_embeddings.py:358](/Users/davidabelson/developer/BacPredict/src/tl/embed/generate_embeddings.py#L358) — Bacformer-contextualised embedding generation.
- Each task package's `train_amr.py` (or equivalent) `ArgumentParser` default for `model_name_or_path`. Currently only [src/kleb_ast/train_amr.py:383](/Users/davidabelson/developer/BacPredict/src/kleb_ast/train_amr.py#L383) — others are about to be added.

The MAG-trained model (`macwiatrak/bacformer-large-masked-MAG`) is kept only for the one-paragraph comparison run described in [src/kleb_ast/CLAUDE.md](/Users/davidabelson/developer/BacPredict/src/kleb_ast/CLAUDE.md) (Kleb sub-step 3). It is **not** the default anywhere else.

Load it with the canonical idiom — see [[bacformer-loading-idiom]].
