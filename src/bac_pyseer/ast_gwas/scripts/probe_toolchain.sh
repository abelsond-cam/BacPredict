#!/usr/bin/env bash
# Step 0 gate: can the GWAS toolchain actually install on this cluster?
#
# src/bac_pyseer/pixi.toml scopes pyseer/ggcat/unitig-caller to [target.linux-64.dependencies],
# which was right when "the HPC" meant CSD3 (x86). Isambard is aarch64 Grace, so the solve has to
# be proven before any compute is sized -- pyseer is pure Python and should resolve, but ggcat is a
# Rust binary and bioconda's aarch64 coverage is the open risk. Precedent for adding the platform:
# src/bacpredict/apps/kleb/pixi.toml already lists linux-aarch64.
#
# Runs on the login node in a few minutes; no SLURM. Reports what is missing and what to do about
# it rather than mutating pixi.toml -- adding linux-aarch64 there is a deliberate, reviewable edit.
#
# Usage:  bash src/bac_pyseer/ast_gwas/scripts/probe_toolchain.sh
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
PIXI_MANIFEST=${PIXI_MANIFEST:-$REPO/src/bac_pyseer/pixi.toml}
export PATH="$HOME/.pixi/bin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

echo "=== platform ==="
echo "arch:     $(uname -m)"
echo "host:     $(hostname)"
echo "manifest: $PIXI_MANIFEST"
grep -A 3 '^platforms' "$PIXI_MANIFEST" 2>/dev/null || grep -n 'platforms' "$PIXI_MANIFEST" || true

status=0
echo
echo "=== tools via pixi ==="
for tool in pyseer ggcat mash; do
    if pixi run --manifest-path "$PIXI_MANIFEST" "$tool" --version </dev/null 2>&1 | head -2; then
        echo "  OK   $tool"
    else
        echo "  MISS $tool"
        status=1
    fi
done

echo
echo "=== fallbacks on PATH (used if pixi cannot provide the tool) ==="
for tool in pyseer ggcat mash pigz; do
    printf '  %-8s %s\n' "$tool" "$(command -v "$tool" || echo '-')"
done

if [ "$status" -ne 0 ]; then
    cat <<'EOF'

=== NOT READY ===
Fix, in order of preference:

1. Add "linux-aarch64" to `platforms` in src/bac_pyseer/pixi.toml and re-solve:
       pixi install --manifest-path src/bac_pyseer/pixi.toml
   Note the lock is already stale relative to the toml (ggcat and unitig-caller do not appear in
   pixi.lock), so a re-solve is needed regardless.

2. If ggcat has no aarch64 conda build, build it from source -- Rust cross-compiles cleanly and
   $HOME/.cargo/bin is already on PATH in these scripts:
       cargo install --git https://github.com/algbio/ggcat --locked ggcat_cmdline

3. If pyseer itself will not solve, that is the blocker: it is pure Python, so a plain venv
   (`uv pip install pyseer`) is the fallback -- but keep it OUT of the repo's uv env, which pins
   Bacformer's torch stack that pyseer's numpy/scipy would fight.

Whatever is used, record it: the environment is technical debt to clean up before publishing.
EOF
    exit 1
fi

echo
echo "=== READY — toolchain resolves on this cluster ==="
