"""Generate baclm-350m-masked mean-pooled embeddings for coding + non-coding regions (GPU).

`macwiatrak/baclm-350m-masked` is a mixed protein+DNA char-level masked LM (960-d). For each
genome we embed BOTH modalities and mean-pool each region into one 960-d vector:

  * coding      — the CDS protein sequences (UPPERCASE amino acids), from
                  `extract_proteins_from_gff_fna` (same flat order as the ESM-C store);
  * non-coding  — the intergenic DNA regions (lowercase nucleotides), from
                  `extract_intergenic_from_gff_fna`.

The "slight input change" the model needs vs a standard encoder: the char-level tokenizer is
**case-sensitive** (UPPERCASE = protein modality, lowercase = DNA) and the forward pass takes
``token_type_ids`` (which the tokenizer derives from case) to tell the two apart. We take
``outputs.last_hidden_state`` and attention-masked-mean-pool over residues. Not per-residue
(deferred) and not fed to Bacformer — a standalone store.

Saves `{sample_id}_baclm_embeddings.pt` (bf16): ``protein_embeddings`` [n_cds, 960],
``intergenic_embeddings`` [n_ig, 960], plus region-count + intergenic-coordinate metadata.
Reads the same `(Sample, sr_assembly_file, sr_gff_file)` CSV as the protein-sequence step.
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
from transformers import AutoModel, AutoTokenizer

from tl.embed.extract_intergenic_from_gff_fna import extract_intergenic_from_gff_fna
from tl.embed.extract_proteins_from_gff_fna import extract_proteins_from_gff_fna

BACLM_MODEL_ID = "macwiatrak/baclm-350m-masked"
MAX_LEN = 2048  # model context length (char-level)

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


@torch.no_grad()
def mean_pool_embeddings(seqs: list[str], model, tokenizer, device: str, batch_size: int) -> torch.Tensor:
    """Embed a homogeneous-modality list of sequences → attention-masked mean-pooled [n, 960] (bf16, CPU).

    ``seqs`` must be one modality (all UPPERCASE protein OR all lowercase DNA) — the tokenizer
    reads modality from case and emits ``token_type_ids`` accordingly. Returns an empty [0, H]
    tensor for an empty input.
    """
    if not seqs:
        hidden = getattr(model.config, "hidden_size", 960)
        return torch.empty((0, hidden), dtype=torch.bfloat16)

    out: list[torch.Tensor] = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i : i + batch_size]
        batch = tokenizer.batch_encode_plus(
            chunk, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt"
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        res = model(
            input_ids=batch["input_ids"],
            token_type_ids=batch.get("token_type_ids"),
            attention_mask=batch.get("attention_mask"),
        )
        hidden = res.last_hidden_state  # [B, L, H]
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # [B, L, 1]
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)  # [B, H]
        out.append(pooled.to(torch.bfloat16).cpu())
    return torch.cat(out, dim=0)


def process_genome(row, model, tokenizer, device: str, output_dir: Path, batch_size: int,
                   min_intergenic_len: int) -> tuple[str, bool, str]:
    """Embed one genome's proteins + intergenic regions and save its baclm .pt."""
    sample_id = str(row["Sample"])
    try:
        gff, fna = str(row["sr_gff_file"]), str(row["sr_assembly_file"])
        # Coding: flatten the per-contig protein lists into ESM-C flat-index order.
        prot = extract_proteins_from_gff_fna(gff, fna)
        proteins = [p for contig in prot["protein_sequence"] for p in contig]
        # Non-coding: intergenic DNA regions (lowercase).
        ig = extract_intergenic_from_gff_fna(gff, fna, min_len=min_intergenic_len)

        prot_emb = mean_pool_embeddings(proteins, model, tokenizer, device, batch_size)
        ig_emb = mean_pool_embeddings(ig["intergenic_sequence"], model, tokenizer, device, batch_size)

        torch.save(
            {
                "protein_embeddings": prot_emb,              # [n_cds, 960] bf16, flat-index order
                "intergenic_embeddings": ig_emb,             # [n_ig, 960] bf16
                "n_proteins": int(prot_emb.shape[0]),
                "n_intergenic": int(ig_emb.shape[0]),
                "intergenic_seqid": ig["intergenic_seqid"],
                "intergenic_start": ig["intergenic_start"],
                "intergenic_end": ig["intergenic_end"],
            },
            output_dir / f"{sample_id}_baclm_embeddings.pt",
        )
        return sample_id, True, ""
    except Exception as e:  # noqa: BLE001
        msg = f"Error processing {sample_id}: {e}"
        logger.error(msg)
        return sample_id, False, msg


def main() -> None:
    """Run baclm coding + non-coding embedding over the input CSV (array-sliceable)."""
    ap = argparse.ArgumentParser(description="baclm-350m-masked mean-pooled coding + non-coding embeddings.")
    ap.add_argument("--input-csv", type=Path, required=True, help="CSV: Sample, sr_assembly_file, sr_gff_file")
    ap.add_argument("--output-dir", type=Path, required=True, help="output dir for {Sample}_baclm_embeddings.pt")
    ap.add_argument("--n", type=int, default=None, help="limit number of genomes (testing)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--min-intergenic-len", type=int, default=30)
    ap.add_argument("--start-idx", type=int, default=None, help="array slice start (over sorted CSV rows)")
    ap.add_argument("--end-idx", type=int, default=None, help="array slice end")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)
    df = df.dropna(subset=["Sample", "sr_assembly_file", "sr_gff_file"]).sort_values("Sample").reset_index(drop=True)
    if args.n:
        df = df.head(args.n)

    # Array slice BEFORE skip-existing (fixed responsibility set; avoids the shrinking-list race).
    if args.start_idx is not None and args.end_idx is not None:
        total = len(df)
        df = df.iloc[args.start_idx : args.end_idx].reset_index(drop=True)
        logger.info(f"Array slice {args.start_idx}:{args.end_idx} of {total} -> {len(df)} rows")

    if args.skip_existing:
        before = len(df)
        df = df[~df["Sample"].apply(lambda s: (args.output_dir / f"{s}_baclm_embeddings.pt").exists())]
        logger.info(f"skip-existing: dropped {before - len(df)} already-done")
    if df.empty:
        logger.info("nothing to do")
        return

    logger.info(f"Loading baclm ({BACLM_MODEL_ID}) on {args.device}...")
    model, tokenizer = load_baclm(args.device)
    logger.info("baclm loaded")

    n_ok = n_fail = 0
    t0 = time.time()
    logger.info(f"START {datetime.now():%Y-%m-%d %H:%M:%S} | {len(df)} genomes")
    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="baclm"), 1):
        sample_id, ok, _ = process_genome(
            row, model, tokenizer, args.device, args.output_dir, args.batch_size, args.min_intergenic_len
        )
        if ok:
            n_ok += 1
        else:
            n_fail += 1
        if idx % 10 == 0:
            per = (time.time() - t0) / idx
            logger.info(f"[{idx}/{len(df)}] {per:.1f}s/genome | ok={n_ok} fail={n_fail} "
                        f"| ETA {timedelta(seconds=int(per * (len(df) - idx)))}")

    logger.info(f"DONE {datetime.now():%Y-%m-%d %H:%M:%S} | ok={n_ok} fail={n_fail} | {timedelta(seconds=int(time.time() - t0))}")


if __name__ == "__main__":
    main()
