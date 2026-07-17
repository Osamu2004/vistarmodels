## Attempt 1 — Stage 1: Initial Implementation

**Hypothesis**: AnySD's official `visual_segment` expert can be mapped to Vistar's shared SECOND one-class protocol by passing the real source image, the selected target-side semantic change mask as its visual prompt, and a class-specific instruction containing the trained `[V*]` placeholder.

**Code Changes**: Added an official-source bootstrap, selective Hugging Face weight download, dependency checker, local-path AnySD loader, distributed manifest runner, SECOND launcher, documentation, and the `anysd` shared-manifest consumer contract.

**Configuration**: Official AnySD visual-segmentation expert; SD 1.5 base; FP16; 512-pixel generation; 100 steps; text guidance 1.5; source-image guidance 2.0; visual-reference scale 0.8; shared one-class selection seed 42.

**Result**: Python AST parsing, command-line help, Bash syntax checks, bootstrap no-download mode, dependency failure reporting, and `git diff --check` pass. Full CUDA generation and benchmark metrics require the Linux GPU environment and official model download.

**Analysis**: The wrapper deliberately loads only one AnySD pipeline instead of the official demo's duplicated general and expert pipelines. **[Reusable]** Selectively downloading a single task expert and constructing the official pipeline from local component paths reduces storage and avoids hidden Hugging Face cache dependencies while preserving official inference semantics.

## Attempt 2 — Full Multi-Class Visual Condition

**Hypothesis**: AnySD's visual-segmentation expert should receive the complete target-side directional change mask rather than a randomly selected single category, because the full color layout preserves simultaneous semantic transitions and matches VISTAR's multi-class generation setting.

**Code Changes**: Added an AnySD-specific SECOND manifest builder that retains all changed categories, changed the default launcher protocol to `full_multiclass`, synchronized the fixed seven-color palette with direction-aware class text, made seeds stable per sample and direction, and preserved the former one-class path behind `MASK_MODE=oneclass`.

**Configuration**: Source-time image + complete target-side semantic color mask + class-aware `[V*]` instruction; SECOND official palette; prompt vocabulary `changed inland water`, `changed bare land`, `changed grass`, `changed forest`, `changed building`, and `changed playground`; both temporal directions by default.

**Validation**: AST compilation, Python bytecode compilation, Bash syntax checking, and `git diff --check` pass. A synthetic bidirectional SECOND pair verifies that a T1-to-T2 mask retains classes 4 and 5 together, the reverse mask retains classes 1 and 2 together, and the direction-specific prompt remains below the CLIP text limit. Full CUDA inference remains to be run in the Linux AnySD environment.
