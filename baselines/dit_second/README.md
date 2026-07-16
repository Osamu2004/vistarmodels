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

## Train on two GPUs

The recommended one-command entry automatically bootstraps DiT, creates or
reuses the online train manifest, runs the two-step/resume smoke test, and then
starts the resumable 300K training run:

```bash
bash run_bash/dit_b2_second_oneclick.bash
```

Its defaults match the WSL host paths used for this baseline. Set
`RUN_FULL=0` for smoke-only, `RUN_SMOKE=0` to skip an already completed smoke
test, `REBUILD_MANIFEST=1` to recreate the path-only manifest, or
`INSTALL_DEPS=1` to install `requirements-dit.txt` before launch.

The lower-level training entry remains available:

```bash
VAE_MODEL=/root/data/weight/stable-diffusion-v1-5 \
GPU_IDS=0,1 NPROC_PER_NODE=2 \
bash run_bash/dit_b2_second_train.bash
```

Defaults are 256x256, per-GPU batch 4, no gradient accumulation, and global
batch 8 on two GPUs. Training uses AdamW at 1e-4, no weight decay, EMA 0.9999, bf16, horizontal
flip probability 0.5, and 300K optimizer updates. `RESUME=auto` loads the
largest `checkpoint-*.pt` in the output directory. Every run writes
`train_config.json`, `train_log.jsonl`, `latest.json`, and complete model/EMA/
optimizer/scheduler/scaler/RNG checkpoints.

Distributed training uses the Gloo process-group backend, matching VISTAR.
The launcher rejects non-Gloo backends so a resumed run cannot silently fall
back to NCCL.

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
