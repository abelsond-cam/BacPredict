# API

> **⚠ This reference is stale and is not currently generated.**
>
> It documented `tl.embed`, `tl.genome_download`, `tl.train`, `tb_ast` and `kleb_ast` — a module
> layout that **stopped existing on 2026-07-11**, when the parallel task packages were consolidated
> into one engine plus thin per-organism apps. Every `automodule` directive it carried pointed at an
> unimportable module, so the page could not build.
>
> Rather than leave directives that document modules nobody can import, the page is reduced to this
> pointer until someone needs a generated API reference enough to rebuild it against the real tree.

## Where the code actually is

| Was documented as | Now |
|---|---|
| `tl.embed` | `bacpredict.engine.embedding` |
| `tl.genome_download` | `bacpredict.engine.download` |
| `tl.train` (`split_utils`, `datasets`) | `bacpredict.engine.splits` and `bacpredict.engine.finetune` |
| `tb_ast`, `kleb_ast` | `bacpredict.engine` + `bacpredict.apps.{tb,kleb}` |
| `pangena_predict` | `bacpredict.engine.gene_lr` and `bacpredict._archive` |
| `kleb_iso_source` | unchanged — still `src/kleb_iso_source/` |

The authoritative map, including the full dead-path table, is
[`PROJECT_STATE.md`](../PROJECT_STATE.md) §2. Package-level orientation is in the root
[`CLAUDE.md`](../CLAUDE.md) under *Package layout*.

## Rebuilding this page

Point `automodule` at the engine subpackages listed in `PROJECT_STATE.md` §2 and at the two app
packages. Check each target imports cleanly (`uv run python -c "import bacpredict.engine.splits"`)
before adding a directive — an unimportable target is what left this page broken and unnoticed.
