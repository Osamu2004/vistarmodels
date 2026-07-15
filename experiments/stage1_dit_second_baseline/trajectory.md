# Stage 1: Source+Mask DiT-B/2 Baseline Reproduction

- **Pipeline**: official DiT-B/2 adapted for directional SECOND target-image generation
- **Budget**: 20 attempts
- **Gate**: complete training converges, the 3,388-target bidirectional SECOND test run finishes, and common Table 4 metrics are computed
- **Start date**: 2026-07-15

## Attempt 1 — Reproducible training and inference implementation

**Hypothesis**: A DiT-B/2 whose patch projection consumes the noisy target
latent together with frozen-VAE source-image and directional-mask latents gives
a controlled in-domain baseline with the same high-level `source image + mask`
condition contract as VISTAR.

**Code Changes**: Replaced the prototype loop with torchrun/DDP training,
`DistributedSampler`, optimizer-step-based gradient accumulation, bf16/fp16/fp32
handling, per-update EMA, atomic full-state checkpoints, automatic resume,
deterministic sampling, manifest validation, structured JSON logs, pinned
official DiT bootstrap, dependency checks, and VISTAR Table 4 output export.
The dataset no longer returns variable-length `changed_class_ids` to the default
collator, and RGB semantic masks use nearest-neighbor resizing.

**Configuration**: Official DiT-B/2, 256x256 RGB images and masks, SD 1.5 VAE,
12-channel patch input (4 noisy target + 4 source + 4 mask), eight-channel
epsilon/sigma output, dummy one-class embedding without dropout, AdamW at
`1e-4`, zero weight decay, EMA `0.9999`, per-GPU batch 2, two GPUs, gradient
accumulation 4 (global batch 16), bf16, random horizontal flip `p=0.5`, seed 42,
300K optimizer updates. Inference defaults to EMA, 250 diffusion steps, no CFG,
and deterministic per-sample noise.

**Result**: Python byte-compilation, Bash syntax validation, whitespace
validation, and train/inference launcher dry-runs pass locally. The bootstrap
script successfully cloned and checked out official DiT revision
`ed81ce2229091fd4ecc9a223645f95cf379d582b` under `/tmp`, and direct inspection
confirmed that upstream `training_losses` and `p_sample_loop` forward arbitrary
`model_kwargs` to the model as required by the spatial `condition` argument. A
CPU synthetic self-test also passed: two manifest rows whose omitted
`changed_class_ids` lengths differed collated into `[2,3,64,64]`, the adapted
patch projection accepted 12 channels, and a reduced-spatial-size DiT-B/2
forward returned `[1,8,8,8]` with 129,572,384 trainable parameters. A CUDA smoke
test and the full SECOND training/evaluation have not been run on this macOS
workspace because it has no CUDA device or SD 1.5 VAE snapshot.

**Analysis**: Static Stage 1 integration is complete, but the metric gate remains
open. The first Linux/CUDA action should be a two-optimizer-step smoke run,
followed by a resume test from `checkpoint-0000002.pt`; only then should the
300K run start. The current baseline uses the complete directional semantic
change mask. If Table 4 is later changed to a shared one-class condition
protocol, the train and test manifests must both be regenerated under that same
policy rather than altering only inference.
