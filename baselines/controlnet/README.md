# SD 1.5 ControlNet for SECOND

This baseline trains a ControlNet from the local Stable Diffusion 1.5 snapshot
on both directions of SECOND.  Each example conditions the generator on the
target-side directional semantic change mask and a class-aware English prompt;
the source-time image is exported for evaluation but is not passed to
ControlNet.

For example, a T1-to-T2 row containing building and playground changes uses:

```text
Generate a realistic 256 by 256 post-change remote sensing image. The
target-side semantic change mask contains changed building and changed
playground; unchanged pixels are labeled unchanged.
```

The reverse row uses `pre-change` and derives its changed-class list from the
reverse target mask.  The class vocabulary is `inland water`, `bare land`,
`grass`, `forest`, `building`, and `playground`.

## One-command training

The default run is intentionally one GPU with batch size 2, no gradient
accumulation, two DataLoader workers, bf16, 256x256 inputs, and 100K optimizer
steps.  It first builds path-only train/test manifests, verifies the local
weights and data, runs a two-step plus resume smoke test, and then starts or
resumes the full run.

```bash
cd /root/code/vistarmodels

SECOND_ROOT=/root/data/second_dataset \
BASE_MODEL=/root/data/weight/stable-diffusion-v1-5 \
GPU_IDS=0 \
NPROC_PER_NODE=1 \
PER_GPU_BATCH=2 \
bash run_bash/controlnet_second_oneclick.bash
```

If the active environment lacks the pinned packages, add `INSTALL_DEPS=1` to
that command or install them once with:

```bash
python -m pip install -r requirements-controlnet.txt
```

The default full output is:

```text
/root/data/experiment/controlnet_sd15_second_mask_text_256_1gpu_bs2_seed42
```

`RESUME=auto` is the default.  To continue the full run without repeating the
smoke test:

```bash
cd /root/code/vistarmodels

GPU_IDS=0 NPROC_PER_NODE=1 PER_GPU_BATCH=2 \
RUN_SMOKE=0 RUN_FULL=1 RESUME=auto \
bash run_bash/controlnet_second_oneclick.bash
```

Checkpoints are full Accelerate states under `checkpoint-XXXXXXX`; the final
directory also contains a Diffusers `save_pretrained` ControlNet.

## Generation and common SECOND metrics

The inference exporter writes `source_rgb`, `cond_mask`,
`cond_mask_official`, `cond_mask_ids`, `gt_rgb`, `pred_rgb`, `absdiff`, and
`prompts`, plus JSONL/config metadata.  This is the same saved-output contract
used by the other SECOND generation baselines.

```bash
cd /root/code/vistarmodels

CONTROLNET=/root/data/experiment/controlnet_sd15_second_mask_text_256_1gpu_bs2_seed42 \
BASE_MODEL=/root/data/weight/stable-diffusion-v1-5 \
GPU_IDS=0,1 NPROC_PER_NODE=2 \
COMPUTE_METRICS=1 \
bash run_bash/controlnet_second_gen.bash
```

For single-GPU inference, set `GPU_IDS=0 NPROC_PER_NODE=1`.  The common metric
stage reports mFD6, FID-Sat, PSNR, SSIM, LPIPS, CMMD, and frozen directional
change-consistency metrics when the VISTAR evaluator and detector checkpoint
are available.

For paper comparisons, label the condition as **change mask + text**.  It is
not a source-image-conditioned editing baseline.
