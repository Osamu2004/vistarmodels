# DiT-B/2 conditioned SECOND baseline

This baseline trains the official `facebookresearch/DiT` DiT-B/2 architecture
for directional SECOND target-image generation. Each training instance is:

```text
source RGB + target-time directional semantic-change mask -> target RGB
```

The frozen Stable Diffusion 1.5 VAE encodes the target, source, and RGB mask.
The noisy four-channel target latent is concatenated with the four-channel
source latent and four-channel mask latent before DiT's patch projection. The
model retains DiT's eight-channel epsilon/sigma output, so only the input patch
projection changes from 4 to 12 channels. The dummy class embedding has no
dropout and no classifier-free guidance is used.

## Prepare

```bash
python -m pip install -r requirements-dit.txt
bash scripts/bootstrap_dit.sh
SECOND_ROOT=/root/data/second_dataset SECOND_SPLITS=train \
bash run_bash/dit_b2_second_prepare.bash
```

The DiT preparation command writes only a lightweight bidirectional JSONL. It
does not copy or resize the source/target RGB images. Raw SECOND labels are
decoded, resized with nearest-neighbor interpolation, and rendered with the
canonical palette online in each DataLoader worker. The default manifest is:

```text
/root/data/experiment/dit_b2_second_data/second/train.jsonl
```

To prepare the test manifest later without deleting the train manifest:

```bash
SECOND_ROOT=/root/data/second_dataset SECOND_SPLITS=test \
bash run_bash/dit_b2_second_prepare.bash
```

## Resume training on one GPU

The recommended one-command entry automatically bootstraps DiT, creates or
reuses the online train manifest, and resumes the 300K training run from the
newest full-state checkpoint in the output directory:

```bash
bash run_bash/dit_b2_second_oneclick.bash
```

Its defaults match the WSL host paths used for this baseline. The completed
smoke test is skipped by default; set `RUN_SMOKE=1 RUN_FULL=0` to rerun it,
`REBUILD_MANIFEST=1` to recreate the path-only manifest, or
`INSTALL_DEPS=1` to install `requirements-dit.txt` before launch.

The lower-level training entry remains available:

```bash
VAE_MODEL=/root/data/weight/stable-diffusion-v1-5 \
GPU_IDS=0 NPROC_PER_NODE=1 RESUME=auto \
bash run_bash/dit_b2_second_train.bash
```

Defaults are 256x256, single-GPU batch 4, no gradient accumulation, global batch
4, and two DataLoader workers. Training uses AdamW at 1e-4, no weight decay,
EMA 0.9999, bf16, horizontal
flip probability 0.5, and 300K optimizer updates. `RESUME=auto` loads the
largest `checkpoint-*.pt` in the output directory. Every run writes
`train_config.json`, `train_log.jsonl`, `latest.json`, and complete model/EMA/
optimizer/scheduler/scaler/RNG checkpoints.

If a checkpoint was written by the previous two-GPU run, resume converts its
saved `next_batch_in_epoch` using the number of globally processed samples so
that the single-GPU loader starts at the corresponding epoch position. Model,
EMA, optimizer, scheduler, scaler, optimizer step, and the available rank-0 RNG
state are restored. Reducing the worker count changes worker-local augmentation
RNG streams, so bitwise-identical data augmentation is not guaranteed.

Multi-GPU training uses the Gloo process-group backend, matching VISTAR. With
the new single-GPU default no gradient collective is initialized; the launcher
still rejects non-Gloo overrides for a consistent interface.

By default only the newest three periodic checkpoints are retained; set
`--keep_last 0` to keep every checkpoint.

For a short infrastructure test, append `MAX_STEPS=2 NUM_WORKERS=0` before the
launcher. This is not a baseline result.

## Generate Table 4 outputs

```bash
VAE_MODEL=/root/data/weight/stable-diffusion-v1-5 \
CHECKPOINT=/root/data/experiment/dit_b2_second_source_mask_256_seed42/checkpoint-0300000.pt \
bash run_bash/dit_b2_second_gen.bash
```

Inference defaults to EMA weights, 250 diffusion steps, deterministic per-name
noise, and posterior-mode source/mask VAE latents. It writes the common VISTAR
Table 4 directory contract: `source_rgb`, `cond_mask`, `cond_mask_official`,
`cond_mask_ids`, `gt_rgb`, `pred_rgb`, `absdiff`, and `prompts`.
