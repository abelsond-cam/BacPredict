#!/usr/bin/env bash
# Where does baclm per-genome time actually go? Time the 4 phases separately for a few genomes:
# protein extraction (CPU), intergenic extraction (CPU), protein embedding (GPU), IGR embedding
# (GPU). The budget sweep showed bigger GPU batches were SLOWER and mem stayed <41GB — implying the
# bottleneck is CPU extraction, not the GPU. This confirms/refutes that before we optimise anything.
#SBATCH --job-name=diagnose-baclm-phases
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
set -uo pipefail
: "${SCRATCHDIR:?}"
S="$SCRATCHDIR"; PY="$S/envs/bacpredict-gpu-venv/bin/python"
export HF_HOME="$S/cache/hf" TORCH_HOME="$S/cache/torch"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
cd "$S/tmp"

"$PY" - "$S" <<'PY'
import sys, time, os
sys.path.insert(0, os.path.expanduser("~/BacPredict/src"))
import pandas as pd, torch
from tl.embed.baclm_embed import load_baclm, mean_pool_embeddings
from tl.embed.extract_proteins_from_gff_fna import extract_proteins_from_gff_fna
from tl.embed.extract_intergenic_from_gff_fna import extract_intergenic_from_gff_fna
S = sys.argv[1]
df = pd.read_csv(f"{S}/processed/smoke_kleb/input_smoke.csv").head(3)
print("loading baclm..."); model, tok = load_baclm("cuda:0")
def clk(fn):
    torch.cuda.synchronize(); t=time.time(); r=fn(); torch.cuda.synchronize(); return r, time.time()-t
for _, r in df.iterrows():
    g, f = r["sr_gff_file"], r["sr_assembly_file"]
    prot, t_pe = clk(lambda: extract_proteins_from_gff_fna(g, f))
    proteins = [p for c in prot["protein_sequence"] for p in c]
    ig, t_ie = clk(lambda: extract_intergenic_from_gff_fna(g, f))
    _, t_pg = clk(lambda: mean_pool_embeddings(proteins, model, tok, "cuda:0", 12288))
    _, t_ig = clk(lambda: mean_pool_embeddings(ig["intergenic_sequence"], model, tok, "cuda:0", 12288))
    print(f"{r['Sample']}: n_prot={len(proteins)} n_ig={len(ig['intergenic_sequence'])} "
          f"|| prot_extract={t_pe:.1f}s  ig_extract={t_ie:.1f}s  prot_GPU={t_pg:.1f}s  ig_GPU={t_ig:.1f}s "
          f"|| total={t_pe+t_ie+t_pg+t_ig:.1f}s  CPU={t_pe+t_ie:.1f}s  GPU={t_pg+t_ig:.1f}s")
PY
echo "=== diagnostic done ==="
