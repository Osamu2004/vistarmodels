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
`1e-4`, zero weight decay, EMA `0.9999`, per-GPU batch 4, two GPUs, no gradient
accumulation (global batch 8), bf16, random horizontal flip `p=0.5`, seed 42,
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

## Attempt 2 — Online native SECOND conversion

**Hypothesis**: Keeping source/target RGB and semantic labels at their native
SECOND paths while performing resize, label decoding, and palette rendering in
DataLoader workers should produce exactly the same tensors as the materialized
cache without duplicating the dataset on disk.

**Code Changes**: Added an online manifest mode with raw source, target, and
direction-specific target-label paths; extended the DiT dataset to decode
grayscale IDs or known RGB palettes online; added online inference export of
canonical ID/RGB masks; introduced `dit_b2_second_prepare.bash`; and changed the
DiT train/test manifest defaults to `/root/data/experiment/dit_b2_second_data`.
The shared preparation path retains its original materialized mode for
ControlNet, SPADE, OASIS, and other consumers.

**Configuration**: Same directional protocol and 256x256 training resolution as
Attempt 1. Online mode performs bicubic source/target resize, nearest-neighbor
label resize, IDs 0--6 validation, and canonical SECOND palette rendering per
sample. The manifest itself is the only persistent prepared dataset artifact.

**Result**: A synthetic two-scene native SECOND tree produced four bidirectional
online rows and zero copied PNGs. A batch loaded from online paths had source
and mask shapes `[2,3,64,64]`; its mask tensor was exactly equal to the legacy
materialized-mode tensor. Online inference export reconstructed a complete
Table 4 output record with canonical `cond_mask_ids` and RGB mask. Python
compilation, Bash syntax, and whitespace checks pass.

**Analysis**: Hypothesis confirmed. **[Reusable]** When preprocessing is fully
deterministic and inexpensive relative to model training, a path-only manifest
plus online canonicalization avoids redundant dataset copies while preserving
the controlled input tensor contract. The CUDA smoke/resume and metric gates
remain open.

## Attempt 3 — One-command guarded launch

**Hypothesis**: A single launcher that performs environment checks, pinned-source
bootstrap, online manifest creation, smoke/resume validation, and full training
in a fixed order will reduce manual path/configuration errors without weakening
the Stage 1 gate.

**Code Changes**: Added `dit_b2_second_oneclick.bash`. It defaults to the user's
WSL SECOND/VAE paths and two GPUs, verifies the active Python and VAE layout,
bootstraps pinned DiT, rebuilds only missing/non-online manifests, runs optimizer
steps 1--2 and resumes to step 3, then starts or resumes the 300K full run. It
exposes smoke-only/full-only, manifest rebuild, dependency installation, batch,
worker, GPU, output, and dry-run controls.

**Configuration**: Same model and training configuration as Attempts 1--2. The
one-click orchestration defaults to both smoke and full stages; full training
starts only if every preceding command exits successfully.

**Result**: Bash syntax and whitespace checks pass. A no-GPU orchestration test
with the synthetic native SECOND tree successfully validated paths, reused the
pinned official source, created a four-row online manifest with zero PNG copies,
counted its records, and exited cleanly with smoke/full disabled. A full dry-run
expanded the two-step smoke, step-three resume, and 300K full commands with the
expected two-GPU/global-batch settings and forwarded training overrides.

**Analysis**: Static orchestration hypothesis confirmed. The launcher changes
orchestration only and does not alter model inputs, loss, optimizer, EMA, or
sampling protocol. The actual CUDA smoke/resume and full-training gates remain
open.

## Attempt 4 — Replace repeated NCCL synchronization failures with Gloo

**Hypothesis**: Replacing the default NCCL process group with the Gloo backend
used by VISTAR will avoid the reproducible single-node two-GPU NCCL collective
hang while preserving the same DDP model, optimizer, data, and checkpoint
states.

**Failure Cases**: The full run first hung at optimizer step 104,413 when Rank
1's gradient `ALLREDUCE` timed out after 600 seconds. `RESUME=auto` restored
`checkpoint-0100000.pt`, but the run reproduced the same failure near steps
107,061--107,062: Rank 1 timed out on a 1,204,256-element `ALLREDUCE`, aborted,
and torch elastic terminated Rank 0. Neither supplied log includes a CUDA OOM
or an earlier Python exception.

**Code Changes**: `train_dit_second.py` now initializes the default process
group from an explicit `--dist_backend` whose only accepted value is `gloo`.
Both the lower-level and one-click launchers default, validate, log, and forward
`DIST_BACKEND=gloo`. The default process group therefore carries DDP gradient
synchronization, barriers, scalar loss reduction, and checkpoint RNG-object
gathering without NCCL. The backend is written to `train_config.json` through
the parsed training arguments.

**Configuration**: All experiment settings remain unchanged: two GPUs,
per-GPU batch 4, accumulation 1, global batch 8, bf16, AdamW at `1e-4`, EMA
`0.9999`, 256x256 inputs, seed 42, and `RESUME=auto` from the newest retained
full-state checkpoint. Only the distributed backend changes from NCCL to Gloo.

**Result**: Targeted Python byte-compilation, Bash syntax checks, whitespace
checks, and launcher dry-run validation pass locally. A remote CUDA/Gloo
optimizer step and long-run stability test remain pending.

**Analysis**: This is a single-variable infrastructure correction motivated by
two matching NCCL failures. It does not establish that NCCL itself was the
ultimate hardware/driver root cause, but it removes the failing communication
path exactly as requested. The next gate is to resume from the latest retained
checkpoint, verify `dist_backend=gloo` in both launch output and
`train_config.json`, and confirm progress through at least the next periodic
checkpoint.

## Attempt 5 — Single-GPU continuation with fewer workers

**Hypothesis**: Continuing the existing full-state run on one GPU avoids all
inter-GPU gradient collectives, while a global batch of 4 follows the user's
updated batch decision and two DataLoader workers reduce host-side contention.

**Code Changes**: Both launchers now default to GPU 0, one process, per-GPU
batch 4, no gradient accumulation, two workers, and `RESUME=auto`. The one-click
launcher skips the already completed smoke stage by default and enters the full
continuation directly. Checkpoints now record world size and global batch. For
older checkpoints, the trainer infers saved world size from the number of RNG
states and converts `next_batch_in_epoch` through the saved globally processed
sample count before applying it to the one-GPU DataLoader.

**Configuration**: DiT-B/2 and all model/loss/optimizer settings remain
unchanged. The continuation uses one GPU, micro-batch 4, accumulation 1, global
batch 4, two workers, bf16, 300K target optimizer steps, and the existing
`/root/data/experiment/dit_b2_second_source_mask_256_seed42` output directory.

**Result**: Local Python byte-compilation, Bash syntax checks, whitespace
checks, and a one-click dry-run pass. The dry-run expands GPU 0, one process,
micro-batch 4, accumulation 1, global batch 4, two workers, `RESUME=auto`, no
smoke stage, and the existing 300K full-run output directory. CUDA continuation
from the remote checkpoint remains pending.

**Analysis**: The converted batch offset preserves the number of samples
consumed within the saved epoch, but the requested global batch changes from 8
to 4 after resume. Changing worker count also changes worker-local random
augmentation streams, so the continuation preserves checkpoint state and data
position rather than the previous optimization regime or bitwise-identical
sample augmentations.
