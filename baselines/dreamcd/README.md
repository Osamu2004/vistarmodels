# DreamCD Baseline

This wrapper runs the official DreamCD SECOND synthesis code in the same
manifest-to-output style used by the other `vistarmodels` baselines.

Official DreamCD repository:

- <https://github.com/tangkai-RS/DreamCD>
- Weights: <https://huggingface.co/tangkaii/DreamCD>
- Paper: *DreamCD: A Change-Label-Free Framework for Change Detection via a
  Weakly Conditional Semantic Diffusion Model in Optical VHR Imagery*, JAG 2026.

The required official inference source is vendored under
`third_party/DreamCD` at upstream commit
`d4750ff6f7d35fe9640059d7b9cdfe6902fcf9c5`. A fresh clone of `vistarmodels`
therefore does not need a separate DreamCD source clone.

## What This Baseline Is

DreamCD is a closed-set, SECOND/LsSCD-style change image synthesis model. It is
not open-set and it is not a text-driven open-vocabulary model. Its inference
condition is:

```text
source image A + dense pseudo-semantic mask A + dense pseudo-semantic mask B
+ explicit-or-derived binary change mask + source-image AdaIN style -> synthetic image B
```

The official demo uses the paired real target image B for AdaIN style transfer.
This wrapper keeps AdaIN enabled but uses each record's source image instead:
`t1_to_t2` uses that sample's T1 image, while `t2_to_t1` uses that sample's T2
image. The paired target B remains available only to populate Vistar's `gt_rgb`
directory for metric computation and is never passed to the official dataset.

DreamCD's raw binary-mask contract is **255 = changed, 0 = unchanged**.  This
is the convention consumed by the official loader and its DDIM source-copy
mask. An explicit `bcd_mask` is preferred. When it is unavailable, the wrapper
derives the raw BCD from the sparse SECOND target-change map when available,
otherwise from `dense_pseudo_mask_A != dense_pseudo_mask_B`.

Important: standard SECOND `label1/label2` are sparse directional *semantic
change* labels, not dense T1/T2 semantic segmentation masks. They cannot be
passed as DreamCD's `mask_A/mask_B`. The adapter requires DreamCD-compatible
dense pseudo masks for model inference, while it uses the SECOND target-change
label separately to write VISTAR's official `cond_mask_ids`.

For a fair report, describe this as a SECOND-trained closed-set DreamCD baseline
with dense pseudo-semantic masks, an explicit-or-derived binary change mask,
and same-sample source-image AdaIN conditioning. It is not the official
paired-target AdaIN protocol.

## Bootstrap

```bash
cd /root/code/vistarmodels
pip install -r requirements-dreamcd.txt
bash scripts/bootstrap_dreamcd.sh
python tools/check_dreamcd_deps.py
```

`taming-transformers` must remain an editable source checkout. Its current
upstream layout can produce a successful but effectively empty wheel under
modern pip, which leads to `No module named 'taming'`. The pinned editable Git
entry in `requirements-dreamcd.txt` avoids that failure. If a non-editable copy
was already installed, repair it with:

```bash
python -m pip uninstall -y taming-transformers
python -m pip install --no-deps -e \
  'git+https://github.com/CompVis/taming-transformers.git@3ba01b241669f5ade541ce990f7650a3b8f65318#egg=taming-transformers'
```

The requirements file likewise installs the official OpenAI CLIP source rather
than the unrelated package that can be obtained from `pip install clip`.

`bootstrap_dreamcd.sh` verifies the vendored official source and downloads the
official SECOND weights from HuggingFace by default. It retains a clone fallback
only for older checkouts where `third_party/DreamCD` is absent:

```text
/root/data/weight/dreamcd/second/vqvae.ckpt
/root/data/weight/dreamcd/second/ldm.ckpt
```

Use `DREAMCD_DOWNLOAD_WEIGHTS=0` to skip downloading, or pass
`DREAMCD_VQVAE_CKPT` and `DREAMCD_CKPT` to use custom paths. On Windows, the
default folder is visible at:

```text
\\wsl.localhost\Ubuntu-22.04\root\data\weight\dreamcd\second\
```

DreamCD's official environment pins an old CUDA 11.1 PyTorch stack. In practice,
use a separate conda environment and let `tools/check_dreamcd_deps.py` tell you
what is missing.

The checker returns a non-zero exit status unless all inference-critical items
are ready: vendored source, SECOND config, both checkpoints, required Python
packages, official DreamCD imports, and a working CUDA tensor allocation. Run it
on the intended GPU, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/check_dreamcd_deps.py
```

For a CPU-only source/package check, explicitly disable the CUDA requirement:

```bash
DREAMCD_REQUIRE_CUDA=0 python tools/check_dreamcd_deps.py
```

## Build A SECOND Manifest

The manifest builder recognizes the official DreamCD example layout:

```text
img_A/img_B/mask_A/mask_B/bcd_mask
```

and a SECOND tree containing *both* its directional labels and DreamCD's
dense pseudo masks:

```text
im1/im2/label1/label2/mask_A/mask_B/change
```

```bash
python tools/build_dreamcd_second_manifest.py \
  --second_root /root/data/SECOND \
  --split test \
  --direction t1_to_t2 \
  --output /root/data/experiment/dreamcd_second_test_manifest.jsonl
```

If the pseudo masks live outside the SECOND root, pass both
`--pseudo_mask_a_dir` and `--pseudo_mask_b_dir`. A root containing only
`im1/im2/label1/label2` is deliberately rejected: using sparse change labels as
full semantic masks silently corrupts DreamCD's condition. If the tree contains
`bcd_mask`, `change`, or another recognized binary-change folder, it takes
precedence. The manifest automatically records the source image as the AdaIN
style reference for every direction.

Validate the official 255=change convention without loading a checkpoint:

```bash
python tools/check_dreamcd_mask_contract.py --dreamcd_root third_party/DreamCD
```

## Run DreamCD

```bash
python baselines/dreamcd/run_dreamcd_manifest.py \
  --dreamcd_root third_party/DreamCD \
  --manifest /root/data/experiment/dreamcd_second_test_manifest.jsonl \
  --output_dir /root/data/experiment/dreamcd_second_test_gen \
  --ckpt /root/data/weight/dreamcd/second/ldm.ckpt \
  --vqvae_ckpt /root/data/weight/dreamcd/second/vqvae.ckpt \
  --resolution 256 \
  --eval_size 256 \
  --batch_size 4 \
  --ddim_steps 200 \
  --seed 2025 \
  --with_adain
```

One-command SECOND run:

```bash
SECOND_ROOT=/root/data/SECOND \
SPLIT=test \
MAX_SAMPLES=5 \
bash run_bash/dreamcd_second_gen.bash

SECOND_ROOT=/root/data/SECOND \
SPLIT=test \
bash run_bash/dreamcd_second_gen.bash
```

The one-command runner defaults to the Vistar evaluation protocol:
`SPLIT=test` and `DIRECTION=both`. Override either variable only for a targeted
single-split or single-direction run.

Existing `pred_rgb/<name>_pred_rgb.png` files are skipped unless `OVERWRITE=1`
is set, so interrupted runs can be resumed by rerunning the same command.
The default output directory includes `sourceadain_vistar_layout` to distinguish
same-sample source AdaIN from the official paired-target AdaIN protocol.

Before the checkpoint is loaded, the wrapper prepares VISTAR evaluation files
and DreamCD class-ID masks for every manifest record. Both preprocessing and
official sampling display progress bars. The one-command runner keeps internal
masks in a persistent sibling cache at `OUTPUT_DIR.runtime`, outside the final
VISTAR result directory. Existing prediction images still skip diffusion
sampling, but mask preprocessing is deliberately regenerated to invalidate the
old inverted-mask adapter outputs. Set `RUNTIME_DIR=/another/path` to move this
cache.

When `RESOLUTION` equals `EVAL_SIZE` (both default to 256), each completed
DreamCD sample is written directly into `pred_rgb` during sampling. Results are
therefore visible immediately and are preserved for the next resume run even if
the full 3388-sample job is interrupted. A differing evaluation size still uses
the runtime directory for native predictions and resizes them after sampling.

The official runtime `img_B` field points to the same record's source image. It
never points to the paired real target B.

Final outputs use the same SECOND generation directory contract as
`vistar/eval_flux2_second_gen.py`:

```text
output_dir/
  source_rgb/*_source_rgb.png
  cond_mask/*_cond_mask.png
  cond_mask_official/*_cond_mask_official.png
  cond_mask_ids/*_cond_mask_ids.png
  gt_rgb/*_gt_rgb.png
  pred_rgb/*_pred_rgb.png
  absdiff/*_absdiff.png
  prompts/<name>.txt
  class_map.json
  prompt_<direction>_raw.txt
  prompt_<direction>_effective.txt
  manifest.jsonl
  prompts.jsonl
```

When SECOND directional labels are present, `cond_mask*` stores the official
target-class semantic change condition with ID 0 reserved for unchanged pixels.
DreamCD-only native predictions, masks, patched config, CSV, and preview files
are kept in the persistent `OUTPUT_DIR.runtime` sibling directory, not inside
`output_dir`. The generated input manifest is stored inside the result directory
as `output_dir/manifest.jsonl`.
