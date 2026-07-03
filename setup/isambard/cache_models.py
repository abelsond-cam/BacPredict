"""Snapshot-cache the three embedding models to $HF_HOME (persistent, never re-pull).

Run on the CPU build node (has egress) so the paid GPU embedding jobs load from the
local HF cache instead of spending GPU-hours downloading. Pins the ESM-C revision for
byte-parity with BacPredict's production store, records Bacformer's resolved commit,
and adds baclm-350m-masked (the mixed protein+DNA masked LM for the coding + non-coding
embedding runs).

$HF_HOME is shared with nuna (-> $PROJECTDIR/david/cache/hf); ESM-C + Bacformer are
likely already cached there, in which case snapshot_download is a no-op. Constants
mirror src/tl/embed (the parity source of truth).
"""

import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ESMC_MODEL_PATH = "Synthyra/ESMplusplus_small"
ESMC_REVISION = "0c0b9c57a7c3da867c8512176ecddb3922816f80"  # byte-match to prod; main is broken
BACFORMER_MODEL_ID = "macwiatrak/bacformer-large-masked-complete-genomes"
BACLM_MODEL_ID = "macwiatrak/baclm-350m-masked"  # mixed protein+DNA char-level masked LM

HF_HOME = os.environ.get("HF_HOME", "")
print(f"HF_HOME = {HF_HOME!r}")
assert HF_HOME, "HF_HOME must be set (-> $PROJECTDIR/david/cache/hf) so models persist off $HOME"

records = []

# trust_remote_code models ship custom *.py modeling files alongside weights;
# snapshot_download fetches the whole repo (config + modeling code + safetensors).
print(f">>> caching ESM-C {ESMC_MODEL_PATH}@{ESMC_REVISION[:8]}")
print("    ->", snapshot_download(repo_id=ESMC_MODEL_PATH, revision=ESMC_REVISION))
records.append((ESMC_MODEL_PATH, ESMC_REVISION))

for model_id in (BACFORMER_MODEL_ID, BACLM_MODEL_ID):
    print(f">>> caching {model_id} (main)")
    commit = HfApi().model_info(model_id).sha
    print("    ->", snapshot_download(repo_id=model_id, revision=commit), "| commit:", commit)
    records.append((model_id, commit))

# Record the exact revisions cached, beside the HF cache, for reproducibility.
manifest = Path(HF_HOME) / "bacpredict_model_revisions.txt"
manifest.write_text("".join(f"{m}\t{r}\n" for m, r in records))
print("=== model cache complete; revisions recorded at", manifest, "===")
