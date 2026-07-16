"""Generate baclm-350m-masked mean-pooled embeddings for coding + non-coding regions (GPU).

`macwiatrak/baclm-350m-masked` is a mixed protein+DNA char-level masked LM (960-d). For each
genome we embed BOTH modalities and mean-pool each region into one 960-d vector:

  * coding      — the CDS protein sequences (UPPERCASE amino acids), read from the shared
                  ``{sample}_protein_sequences.parquet`` (the same store the ESM-C path consumes);
  * non-coding  — the intergenic DNA regions (lowercase nucleotides), read from
                  ``{sample}_intergenic.parquet`` (written by ``extract_intergenic_to_parquet.py``).

Two-stage split, exactly like the ESM-C pipeline (``generate_embeddings.py``): extraction is a
separate CPU job; this GPU job only *reads* parquet and runs forwards — it never parses a GFF or
translates a CDS. The forward loop mirrors ``bacformer.pp.generate_protein_embeddings``:
length-sort, contiguous ``batch_size``, ``padding="longest"``, ``truncation`` to the model context,
bf16, ``no_grad``, attention-masked mean-pool.

**token_type_ids — the fast path.** baclm's char splitting is fast (Rust), but ``BacLMTokenizer``
overrides ``batch_encode_plus`` to infer ``token_type_ids`` in a *pure-Python per-residue loop*
(``_infer_token_type_ids``), which dominates runtime over ~2M residues/genome. Because we embed each
modality separately, the correct ``token_type_ids`` is a **constant** — protein→0, DNA→1, and every
special token→2 — so we build it with one vectorised op (from ``all_special_ids``) and call the fast
base tokenizer directly, bypassing the Python loop. For a homogeneous batch this is byte-identical
to ``_infer_token_type_ids`` (asserted in the smoke). Not per-residue (deferred) and not fed to
Bacformer — a standalone store.

Saves `{sample_id}_baclm_embeddings.pt` (bf16): ``protein_embeddings`` [n_cds, 960] (unchanged —
the validated coding channel) plus THREE non-coding channels from
``extract_intergenic_from_gff_fna``: ``noncoding_embeddings`` [n_nc, 960] (whole CDS-to-CDS runs =
whole_igr), ``fragment_embeddings`` [n_fr, 960] (runs split at named-feature boundaries), and
``feature_embeddings`` [n_feat, 960] (standalone named non-CDS bodies: rRNA/tRNA/tmRNA/ncRNA/CRISPR/
regulatory_region/oriC), plus coordinate + ``feature_type``/``feature_name`` metadata.

**Long regions are windowed, not truncated.** A non-coding region longer than the model context is split
into ``ceil(L/MAX_LEN)`` **equal** segments (no tiny remainder window — see ``_windowize``), each embedded
+ mean-pooled, then combined by a **token-count-weighted mean** = mean-pooling the whole untruncated region
(``--window-overlap`` > 0 falls back to sliding tiling with boundary context). Proteins are deliberately
left on the plain truncating path so the coding store stays byte-identical.
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

BACLM_MODEL_ID = "macwiatrak/baclm-350m-masked"
MAX_LEN = 2048  # model context length (char-level)
_WINDOW_CHUNK = MAX_LEN - 2  # content budget per window (leaves room for the bos/eos wrap)
_MODALITY = {"protein": 0, "dna": 1}  # token_type_ids: protein=0, DNA=1 (special tokens=2)

# Run with an env that has flash-attn built for GH200 (Maciej's shared bacformer env — see
# embed_baclm.sbatch). flash_attn_varlen makes attention memory linear in total tokens, so batch 128
# peaks at only ~8.5 GiB even at maxlen=2048 and throughput is ~700-800 seq/s. WITHOUT flash-attn
# baclm falls back to a dense "packed SDPA with block-diagonal mask" whose memory is O((batch×len)²)
# — batch 128 then needs ~1 TB and OOMs; a tiny batch avoids the OOM but runs ~100× slower. So this
# default assumes the flash-attn env; length-sorting still keeps padding minimal within each batch.
_DEFAULT_BATCH = 128

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("baclm_embeddings_processing.log")],
)
logger = logging.getLogger(__name__)


def load_baclm(device: str, dtype=torch.bfloat16):
    """Load baclm-350m-masked (+ tokenizer) on ``device`` in eval mode; ``(model, tokenizer)``."""
    tok = AutoTokenizer.from_pretrained(BACLM_MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(BACLM_MODEL_ID, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
    return model, tok


def _encode_fast(tokenizer, seqs: list[str]):
    """Tokenise a batch the way ``BacLMTokenizer`` does, minus the per-residue Python loop.

    Wraps each sequence as ``{bos}{seq}{eos}`` and calls the **fast base** ``batch_encode_plus``
    (``PreTrainedTokenizerFast`` — Rust), skipping ``BacLMTokenizer._infer_token_type_ids``. Returns
    the HF ``BatchEncoding`` with ``input_ids``/``attention_mask`` (no ``token_type_ids`` — the
    caller builds those cheaply). ``input_ids`` are byte-identical to the model tokenizer's own.
    """
    wrapped = [f"{tokenizer.bos_token}{s}{tokenizer.eos_token}" for s in seqs]
    return PreTrainedTokenizerFast.batch_encode_plus(
        tokenizer, wrapped, padding="longest", truncation=True, max_length=MAX_LEN, return_tensors="pt"
    )


@torch.no_grad()
def mean_pool_embeddings(
    seqs: list[str], model, tokenizer, device: str, modality: str, batch_size: int = _DEFAULT_BATCH,
) -> torch.Tensor:
    """Embed a homogeneous-modality list of sequences → attention-masked mean-pooled [n, 960] (bf16, CPU).

    ``modality`` is ``"protein"`` (UPPERCASE amino acids) or ``"dna"`` (lowercase nucleotides); it
    fixes ``token_type_ids`` for every real token (protein→0, DNA→1) with special tokens→2, built
    vectorised from ``all_special_ids`` rather than baclm's per-residue Python inference.

    Length-sorts (to minimise padding), then runs contiguous ``batch_size`` slices — a small fixed
    batch (default :data:`_DEFAULT_BATCH`) that stays memory-safe even at the full ``maxlen=2048``
    (see the note on the attention-matrix bound). Each batch: ``padding="longest"``, ``truncation``
    to :data:`MAX_LEN`, bf16 forward, attention-masked mean-pool. Output order is restored to the
    input order. Returns an empty [0, H] tensor for empty input.
    """
    hidden = getattr(model.config, "hidden_size", 960)
    if not seqs:
        return torch.empty((0, hidden), dtype=torch.bfloat16)
    if modality not in _MODALITY:
        raise ValueError(f"modality must be one of {sorted(_MODALITY)}, got {modality!r}")
    mod_id = _MODALITY[modality]
    special_ids = torch.tensor(sorted(tokenizer.all_special_ids), device=device)

    n = len(seqs)
    order = sorted(range(n), key=lambda i: len(seqs[i]))  # length-sort (padding-minimising)
    results: list[torch.Tensor | None] = [None] * n

    for start in range(0, n, batch_size):
        idxs = order[start : start + batch_size]
        enc = _encode_fast(tokenizer, [seqs[k] for k in idxs])
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        # token_type_ids: mod_id for real tokens, 2 for special tokens (matches _infer_token_type_ids).
        is_special = torch.isin(input_ids, special_ids)
        token_type_ids = torch.where(is_special, 2, mod_id)

        out = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # [B, L, H]
        m = attention_mask.unsqueeze(-1).to(h.dtype)  # [B, L, 1]
        pooled = ((h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)).to(torch.bfloat16).cpu()  # [B, H]
        for pos, k in enumerate(idxs):
            results[k] = pooled[pos]
    return torch.stack(results)


def _windowize(seqs: list[str], chunk: int, overlap: int) -> tuple[list[str], list[int], list[float]]:
    """Split each sequence into windows ≤ ``chunk`` for un-truncated embedding.

    Returns a flat list of window substrings, a parallel ``owner`` list (which input sequence each
    window came from), and a parallel ``weight`` list (window length, for the token-count-weighted
    pool). A sequence of length ≤ ``chunk`` yields exactly one window equal to itself, so short
    regions are byte-identical to the non-windowed path.

    For ``overlap == 0`` (the default), a long sequence is split into ``ceil(L/chunk)`` **equal**
    segments (sizes differ by ≤ 1 char) — NOT ``chunk`` + a tiny remainder. A near-empty tail window
    contextualises poorly and, being length-weighted, would still perturb the pooled mean toward its
    impoverished embedding; equal segments keep every window well-sized and the mean balanced across the
    whole region. ``overlap > 0`` (rare) falls back to sliding ``chunk``-sized tiling stepped by
    ``chunk - overlap`` to add boundary context.
    """
    if overlap < 0 or overlap >= chunk:
        raise ValueError(f"overlap must be in [0, {chunk}), got {overlap}")
    windows: list[str] = []
    owner: list[int] = []
    weight: list[float] = []
    for i, s in enumerate(seqs):
        length = len(s)
        if length <= chunk:
            windows.append(s)
            owner.append(i)
            weight.append(float(max(length, 1)))
            continue
        if overlap == 0:
            n = -(-length // chunk)  # ceil(L/chunk) equal segments, each ≤ chunk
            base, rem = divmod(length, n)
            pos = 0
            for w in range(n):
                size = base + (1 if w < rem else 0)
                sub = s[pos : pos + size]
                pos += size
                windows.append(sub)
                owner.append(i)
                weight.append(float(len(sub)))
            continue
        step = chunk - overlap
        for start in range(0, length, step):
            w = s[start : start + chunk]
            if not w:
                break
            windows.append(w)
            owner.append(i)
            weight.append(float(len(w)))
            if start + chunk >= length:
                break
    return windows, owner, weight


@torch.no_grad()
def mean_pool_windowed(
    seqs: list[str], model, tokenizer, device: str, modality: str, batch_size: int,
    *, chunk: int = _WINDOW_CHUNK, overlap: int = 0,
) -> torch.Tensor:
    """Embed sequences with long ones windowed + token-count-weighted-pooled → [n, 960] (bf16, CPU).

    Every window is embedded by :func:`mean_pool_embeddings` (so batching/length-sort/token_type_ids
    are shared with the protein path), then windows of the same input are combined by a length-weighted
    mean. With ``overlap=0`` this weighted mean equals mean-pooling the whole untruncated region;
    sequences ≤ ``chunk`` pass through unchanged. Returns an empty [0, H] tensor for empty input.
    """
    hidden = getattr(model.config, "hidden_size", 960)
    if not seqs:
        return torch.empty((0, hidden), dtype=torch.bfloat16)
    windows, owner, weight = _windowize(seqs, chunk, overlap)
    win_emb = mean_pool_embeddings(windows, model, tokenizer, device, modality, batch_size)  # [n_win, H]
    out = torch.zeros((len(seqs), win_emb.shape[1]), dtype=torch.float32)
    wsum = torch.zeros(len(seqs), dtype=torch.float32)
    win_emb_f = win_emb.to(torch.float32)
    for k, (o, w) in enumerate(zip(owner, weight, strict=True)):
        out[o] += win_emb_f[k] * w
        wsum[o] += w
    out /= wsum.clamp_min(1.0).unsqueeze(1)
    return out.to(torch.bfloat16)


def _flatten_proteins(protein_parquet: Path) -> list[str]:
    """Read a protein-sequence parquet and flatten its per-contig lists into ESM-C flat order."""
    seq_col = pd.read_parquet(protein_parquet)["protein_sequence"].iloc[0]
    return [p for contig in seq_col for p in contig]


def _col(df: pd.DataFrame, name: str, cast) -> list:
    """Read a single-row parquet list column into native python types (safe-loadable .pt)."""
    if name not in df.columns:
        return []
    return [cast(x) for x in df[name].iloc[0]]


def process_genome(sample_id: str, protein_parquet: Path, intergenic_parquet: Path | None,
                   model, tokenizer, device: str, output_dir: Path, batch_size: int,
                   *, window_overlap: int = 0) -> tuple[str, bool, str]:
    """Embed one genome's proteins + the three non-coding views (read from parquet) → baclm .pt.

    Proteins use the plain truncating mean-pool (the coding store is deliberately unchanged). The three
    non-coding channels — ``noncoding`` (whole CDS-to-CDS runs = whole_igr), ``fragment`` (runs split at
    named-feature boundaries), and ``feature`` (named rRNA/tRNA/tmRNA/ncRNA/CRISPR/regulatory_region/oriC
    bodies) — use the windowed pool so long regions are embedded whole (equal-segment) rather than truncated.
    """
    try:
        proteins = _flatten_proteins(protein_parquet)
        nc_seqs = nc_seqid = nc_start = nc_end = []
        fr_seqs = fr_seqid = fr_start = fr_end = []
        feat_seqs = feat_name = feat_type = feat_seqid = feat_start = feat_end = []
        if intergenic_parquet is not None and intergenic_parquet.exists():
            df = pd.read_parquet(intergenic_parquet)
            # Cast coords to native python types — parquet yields numpy scalars, which
            # torch.load(weights_only=True) (the torch>=2.6 default) refuses.
            nc_seqs = _col(df, "noncoding_sequence", str)
            nc_seqid = _col(df, "noncoding_seqid", str)
            nc_start = _col(df, "noncoding_start", int)
            nc_end = _col(df, "noncoding_end", int)
            fr_seqs = _col(df, "fragment_sequence", str)
            fr_seqid = _col(df, "fragment_seqid", str)
            fr_start = _col(df, "fragment_start", int)
            fr_end = _col(df, "fragment_end", int)
            feat_seqs = _col(df, "feature_sequence", str)
            feat_name = _col(df, "feature_name", str)
            feat_type = _col(df, "feature_type", str)
            feat_seqid = _col(df, "feature_seqid", str)
            feat_start = _col(df, "feature_start", int)
            feat_end = _col(df, "feature_end", int)

        prot_emb = mean_pool_embeddings(proteins, model, tokenizer, device, "protein", batch_size)
        nc_emb = mean_pool_windowed(nc_seqs, model, tokenizer, device, "dna", batch_size, overlap=window_overlap)
        fr_emb = mean_pool_windowed(fr_seqs, model, tokenizer, device, "dna", batch_size, overlap=window_overlap)
        feat_emb = mean_pool_windowed(feat_seqs, model, tokenizer, device, "dna", batch_size, overlap=window_overlap)

        torch.save(
            {
                "protein_embeddings": prot_emb,       # [n_cds, 960] bf16, flat-index order (unchanged)
                "noncoding_embeddings": nc_emb,        # [n_nc, 960] bf16, whole CDS-to-CDS runs (whole_igr)
                "fragment_embeddings": fr_emb,         # [n_fr, 960] bf16, runs split at named-feature bounds
                "feature_embeddings": feat_emb,        # [n_feat, 960] bf16, named non-CDS bodies
                "n_proteins": int(prot_emb.shape[0]),
                "n_noncoding": int(nc_emb.shape[0]),
                "n_fragment": int(fr_emb.shape[0]),
                "n_feature": int(feat_emb.shape[0]),
                "noncoding_seqid": nc_seqid,
                "noncoding_start": nc_start,
                "noncoding_end": nc_end,
                "fragment_seqid": fr_seqid,
                "fragment_start": fr_start,
                "fragment_end": fr_end,
                "feature_name": feat_name,
                "feature_type": feat_type,
                "feature_seqid": feat_seqid,
                "feature_start": feat_start,
                "feature_end": feat_end,
            },
            output_dir / f"{sample_id}_baclm_embeddings.pt",
        )
        return sample_id, True, ""
    except Exception as e:  # noqa: BLE001
        msg = f"Error processing {sample_id}: {e}"
        logger.error(msg)
        return sample_id, False, msg


def main() -> None:
    """Run baclm coding + non-coding embedding over the protein-parquet store (array-sliceable)."""
    ap = argparse.ArgumentParser(description="baclm-350m-masked mean-pooled coding + non-coding embeddings.")
    ap.add_argument("--protein-dir", type=Path, required=True, help="dir of {Sample}_protein_sequences.parquet")
    ap.add_argument("--intergenic-dir", type=Path, required=True, help="dir of {Sample}_intergenic.parquet")
    ap.add_argument("--output-dir", type=Path, required=True, help="output dir for {Sample}_baclm_embeddings.pt")
    ap.add_argument("--n", type=int, default=None, help="limit number of genomes (testing)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH,
                    help=f"sequences per forward (default {_DEFAULT_BATCH}; small keeps the attention "
                    "matrix batch×maxlen² memory-safe at maxlen=2048)")
    ap.add_argument("--start-idx", type=int, default=None, help="array slice start (over sorted parquet list)")
    ap.add_argument("--end-idx", type=int, default=None, help="array slice end")
    ap.add_argument("--window-overlap", type=int, default=0,
                    help="overlap (chars) between windows of a long non-coding/RNA region. 0 (default) = "
                    "non-overlapping tiling, whose token-weighted pool equals mean-pooling the whole region.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    protein_files = sorted(args.protein_dir.glob("*_protein_sequences.parquet"))
    if not protein_files:
        logger.error(f"no *_protein_sequences.parquet under {args.protein_dir}")
        sys.exit(1)
    if args.n:
        protein_files = protein_files[: args.n]

    # Array slice BEFORE skip-existing (fixed responsibility set; avoids the shrinking-list race).
    if args.start_idx is not None and args.end_idx is not None:
        total = len(protein_files)
        protein_files = protein_files[args.start_idx : args.end_idx]
        logger.info(f"Array slice {args.start_idx}:{args.end_idx} of {total} -> {len(protein_files)} files")

    def sample_of(p: Path) -> str:
        return p.name.replace("_protein_sequences.parquet", "")

    if args.skip_existing:
        before = len(protein_files)
        protein_files = [
            p for p in protein_files if not (args.output_dir / f"{sample_of(p)}_baclm_embeddings.pt").exists()
        ]
        logger.info(f"skip-existing: dropped {before - len(protein_files)} already-done")
    if not protein_files:
        logger.info("nothing to do")
        return

    logger.info(f"Loading baclm ({BACLM_MODEL_ID}) on {args.device}...")
    model, tokenizer = load_baclm(args.device)
    logger.info("baclm loaded")

    n_ok = n_fail = 0
    t0 = time.time()
    logger.info(f"START {datetime.now():%Y-%m-%d %H:%M:%S} | {len(protein_files)} genomes")
    for idx, protein_parquet in enumerate(tqdm(protein_files, desc="baclm"), 1):
        sample_id = sample_of(protein_parquet)
        intergenic_parquet = args.intergenic_dir / f"{sample_id}_intergenic.parquet"
        _, ok, _ = process_genome(
            sample_id, protein_parquet, intergenic_parquet, model, tokenizer,
            args.device, args.output_dir, args.batch_size, window_overlap=args.window_overlap,
        )
        n_ok += ok
        n_fail += not ok
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()  # release the genome's peak alloc before the next (fights fragmentation)
        if idx % 10 == 0:
            per = (time.time() - t0) / idx
            logger.info(f"[{idx}/{len(protein_files)}] {per:.1f}s/genome | ok={n_ok} fail={n_fail} "
                        f"| ETA {timedelta(seconds=int(per * (len(protein_files) - idx)))}")

    logger.info(f"DONE {datetime.now():%Y-%m-%d %H:%M:%S} | ok={n_ok} fail={n_fail} "
                f"| {timedelta(seconds=int(time.time() - t0))}")
    if args.device.startswith("cuda"):
        logger.info(f"peak GPU mem: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
