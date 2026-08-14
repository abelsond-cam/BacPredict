---
name: tb-drug-us-spellings
description: TB binary_ast.csv uses US drug spellings — rifampin not rifampicin — and CLI defaults must match
metadata: 
  node_type: memory
  type: project
  originSessionId: 965bdc91-22aa-4d38-9677-c15707d973a6
---

The TB AST label CSV (`processed/train_tb_ast/binary_ast.csv`) uses **US drug spellings**, drawn from the EBI AMR records (`raw/tb/ebi_tb_amr_records.csv`). The notable case is **`rifampin`** (US) rather than **`rifampicin`** (UK / WHO).

Full TB drug column list (lower-case, US):
`amikacin, amoxicillin, bedaquiline, capreomycin, cefpimizole, cycloserine, delamanid, ethambutol, ethionamide, isoniazid, kanamycin, levofloxacin, linezolid, moxifloxacin, ofloxacin, para-aminosalicylic acid, pyrazinamide, rifabutin, rifampin, streptomycin`.

**Why:** EBI source uses these spellings. The root [BacPredict_Training_Plan.md](/Users/davidabelson/developer/BacPredict/BacPredict_Training_Plan.md) §1 and [src/tb_ast/CLAUDE.md](/Users/davidabelson/developer/BacPredict/src/tb_ast/CLAUDE.md) reference the **UK** spelling "rifampicin" — that's just docs convention. The CSV column is the source of truth for the training script.

**How to apply:** When wiring up a TB drug for training (`--drug ...`), use the column name as it appears in `binary_ast.csv` — `rifampin`, not `rifampicin`. Same applies to any TB SLURM template, results JSON keying, and stratified-by-mechanism reporting. If a future drug ever needs the UK spelling, fix it at the prepare-script level (rename the column) rather than each downstream consumer.

Related: [[refreshed-bacformer-complete-genomes-model]] for the base model.
