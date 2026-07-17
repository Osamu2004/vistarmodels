# Stage 1 trajectory: SD 1.5 ControlNet on SECOND

## Attempt 1 — direct-manifest implementation

- Date: 2026-07-17
- Goal: establish a reproducible in-domain mask-conditioned change-generation
  baseline for SECOND using the local SD 1.5 snapshot.
- Hypothesis: a ControlNet trained on target-side directional semantic masks
  and class-aware temporal text can synthesize the target-time image under a
  weaker input contract than source-conditioned VISTAR.
- Fixed training defaults: one GPU, per-GPU batch 2, accumulation 1, global
  batch 2, two workers, 256x256, bf16, AdamW, learning rate 1e-5, seed 42, and
  100K optimizer steps.
- Data contract: T1-to-T2 uses T2 as the target and label2 as the condition;
  T2-to-T1 uses T1 as the target and label1 as the condition.  Text says
  `post-change` or `pre-change` accordingly and lists only the target-mask
  classes using the `changed <class>` vocabulary.
- Implementation choice: use the common SECOND JSONL manifest directly.  The
  retired ImageFolder adapter was invalid because only `file_name` was decoded
  as an image while `conditioning_image_file_name` remained a string.
- Static verification: Python byte compilation, Bash syntax, whitespace checks,
  and one-GPU/batch-2 dry-run are required before handoff.
- Synthetic regression: one train scene and one test scene produced two
  directional rows per split.  The forward prompt contained `post-change`,
  `changed building`, and `changed playground`; the reverse prompt contained
  `pre-change` and `changed inland water`.  Online mask decoding returned the
  expected 64x64 ID map.
- Stage-1 gate: pending remote CUDA smoke/resume completion, final
  `save_pretrained` files, 3,388 bidirectional test outputs, and the common
  metric report.  No paper value should be entered before that gate passes.
