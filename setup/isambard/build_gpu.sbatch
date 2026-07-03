#!/usr/bin/env bash
# Build the bacpredict-gpu env on an Isambard-AI compute node (CPU-only job; the install
# is CPU work + model downloads). Run via Slurm because login-node processes are killed
# on SSH-session end (no linger/tmux on Isambard).
#SBATCH --job-name=bacpredict-gpu-build
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
set -uo pipefail

echo "=== node $(hostname) ; egress test ==="
curl -sI --max-time 15 https://pypi.org 2>&1 | head -1 || echo "NO-EGRESS (will need --offline)"

bash "$HOME/BacPredict/setup/isambard/build_gpu.sh"
echo "=== sbatch wrapper done (rc from build_gpu.sh above) ==="
