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

## Attempt 2 — single-process Accelerate initialization repair

- Failure: the remote one-GPU run passed dependency/data checks and then failed
  while constructing `Accelerator`, where `torch.distributed.get_world_size()`
  raised `ValueError: Default process group has not been initialized`.
- Isolation: the launcher correctly used direct Python for one process, so it
  intentionally had no default process group.  The trainer nevertheless passed
  `InitProcessGroupKwargs(backend="gloo")` unconditionally.  In Accelerate
  1.12, that preserves a non-empty backend and reaches `get_world_size()` even
  though no direct-launch process group exists.  Inherited rank variables can
  trigger the same distributed misdetection and remain a secondary risk.
- Cause classification: launcher/environment integration bug, not a model,
  checkpoint, data, CUDA-memory, or optimization failure.
- Fix: the one-process launcher now clears torchrun rank/rendezvous variables
  and passes an explicit `--single_process` guard.  The trainer defensively
  removes the same variables and, critically, does not supply any process-group
  handler to Accelerate in single-process mode.  Multi-process launches still
  receive the requested Gloo handler.  Removed environment values are recorded
  in `train_config.json`.
- Verification: Python byte compilation, Bash syntax, targeted whitespace
  checks, a dry-run seeded with stale rank/world/rendezvous values, and an
  isolated sanitizer unit test pass.  The dry-run remains one GPU, batch 2,
  direct Python, and includes `--single_process`.  The original remote CUDA
  smoke/resume command remains the final runtime check.
