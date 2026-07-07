#!/usr/bin/env bash
# Dependency-free launcher / resumer for the full-cohort embedding arrays (ESM/Bacformer + baclm).
#
# WHY not SLURM --dependency: an `afterok:<extract-job>` dependency is fragile — if extraction
# completes during a login-node outage, its job record ages out and the deferred submit is rejected
# ("Job dependency problem"), stranding the whole run (this cost us ~9 h overnight). Instead this
# script just checks the parquets are on disk and submits directly.
#
# Idempotent + resumable: it submits each array only if a job of that name isn't already queued, and
# the arrays run with --skip-existing, so re-running tops up after a timeout / partial run and never
# duplicates work. Safe to re-run anytime (e.g. after a task hits its wall-clock).
#
# Usage:  ssh <cluster> 'cd $HOME/BacPredict && setup/isambard/submit_embeddings.sh tb|kleb|all'
set -uo pipefail
: "${SCRATCHDIR:?}"
S="$SCRATCHDIR"; REPO="$HOME/BacPredict"
sel="${1:?usage: submit_embeddings.sh tb|kleb|all}"
tasks=("$sel"); [ "$sel" = all ] && tasks=(tb kleb)
declare -A ARR=( [tb]="0-31" [kleb]="0-7" )

queued(){ squeue -u "$USER" -h -o %j | grep -qx "$1"; }
submit_once(){ local name=$1; shift
  if queued "$name"; then echo "  skip $name (already queued)"
  else sbatch -J "$name" "$@" && echo "  submitted $name"; fi; }

for T in "${tasks[@]}"; do
  case "$T" in
    tb)   PROC="$S/processed/train_tb_ast" ;;
    kleb) PROC="$S/processed/train_kleb_ast" ;;
    *) echo "bad task '$T' (want tb|kleb|all)"; exit 1 ;;
  esac
  NP=$(find "$PROC/protein_sequences" -name '*_protein_sequences.parquet' 2>/dev/null | wc -l)
  NI=$(find "$PROC/intergenic" -name '*_intergenic.parquet' 2>/dev/null | wc -l)
  echo "=== $T: protein=$NP intergenic=$NI parquets ==="
  if [ "$NP" -eq 0 ] || [ "$NI" -eq 0 ]; then
    echo "  prep incomplete for $T — run extract_proteins.sbatch first; skipping"; continue
  fi
  submit_once "embed-esmbac-$T" --time=20:00:00 --export=ALL,TASK="$T" --array="${ARR[$T]}" "$REPO/setup/isambard/embed_esm_bacformer.sbatch"
  submit_once "embed-baclm-$T"                   --export=ALL,TASK="$T" --array="${ARR[$T]}" "$REPO/setup/isambard/embed_baclm.sbatch"
done
echo "=== queue ==="
squeue -u "$USER" -o "%.11i %.20j %.10T %R" | head -20
