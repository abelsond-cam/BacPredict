# Parked work

Tasks that were planned in detail, never started, and are not on the current path. They are kept
here — rather than in the state file — because `PROJECT_STATE.md` describes what **is**, and these
describe what **might be**. Nothing here is in flight.

Each file preserves the milestones as they were agreed, so picking one up does not mean re-planning
it from scratch. **The paths and package names in them are from before the July 2026 consolidation**
— check `PROJECT_STATE.md` §2 for where things actually live now.

| File | Task | Why parked |
|---|---|---|
| [`predict_hgt_embedding_diagnostic.md`](predict_hgt_embedding_diagnostic.md) | Does Bacformer preserve HGT identity in its embedding space? | Never started; the package stub was retired in the consolidation |
| [`admixture_detection.md`](admixture_detection.md) | Detect mixed/contaminated assemblies via masked-gene loss | Deferred; needs a design question answered by the Bacformer authors first |

`dp_short_read` (DefensePredictor on short reads) is also deferred but keeps its own stub package —
its milestones live in [`src/dp_short_read/CLAUDE.md`](../../src/dp_short_read/CLAUDE.md).
