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
OUTPUT_DIR=/data/vistar/runs/paper_baselines/data \
bash run_bash/paper_baselines_prepare_data.bash --dataset second
```

The preparation command materializes both temporal directions. Its default
training manifest is:

```text
/data/vistar/runs/paper_baselines/data/second/train.jsonl
```

## Train on two GPUs

```bash
VAE_MODEL=/root/data/weight/stable-diffusion-v1-5 \
GPU_IDS=0,1 NPROC_PER_NODE=2 \
bash run_bash/dit_b2_second_train.bash
```

Defaults are 256x256, per-GPU batch 2, gradient accumulation 4, global batch
16 on two GPUs, AdamW at 1e-4, no weight decay, EMA 0.9999, bf16, horizontal
flip probability 0.5, and 300K optimizer updates. `RESUME=auto` loads the
largest `checkpoint-*.pt` in the output directory. Every run writes
`train_config.json`, `train_log.jsonl`, `latest.json`, and complete model/EMA/
optimizer/scheduler/scaler/RNG checkpoints.

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
