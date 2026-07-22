# vistarmodels

Baselines and wrappers for evaluating generation models against Vistar/UniGen.

This repository is intentionally separate from the main `vistar` codebase. It
keeps external comparison methods reproducible without mixing their dependencies
and source code into the training repository.

New candidate readiness:

| Candidate | Table | Public task checkpoint | Prepared state |
|---|---:|---|---|
| RSEdit UNet (DGTRS-CLIP) | 4 | Yes | Direct SECOND inference |
| ChangeBridge | 4 | No | Data/config/train/sample chain; blocked by missing upstream `runners/` |
| TODSynth | 3 | No | LoveDA chain prepared; blocked by gated SD3.5 Medium access |
| TODSynth + CRFM | 3 | No | Chain prepared; needs gated SD3.5 Medium and a LoveDA MMSeg checkpoint |
| SyntheticGen | 3 | Yes | Released weights cached locally; exact-mask LoveDA Val adapter |
| DDPM | 3 | No | Diffusers train/full inference chain |
| SPADE | 4 | No | Official source + SECOND train/full inference chain |
| OASIS | 4 | No | Official source + SECOND adapter/train/full inference chain |
| ControlNet SD 1.5 | 4 | No | Official Diffusers train/full inference chain |
| ControlNet SD 2.1 | 4 | No | Same chain; gated base-model access still required |
| DiT-B/2 | 4 | No | Official DiT backbone + source/mask latent-conditioning train/inference chain |

SemGAN, SatSynth, Changen, and Changen2 remain hard upstream blocks because
their generator training source or task checkpoints are not public. Prepare
the shared data once, then check the local sources, weights, and inputs:

```bash
bash run_bash/paper_baselines_prepare_data.bash
python tools/check_paper_baseline_readiness.py
```

The readiness checker accepts `PAPER_BASELINE_DATA_ROOT`,
`PAPER_BASELINE_WEIGHT_ROOT`, `PAPER_BASELINE_EVAL_ROOT`, and `HF_HOME` so it
does not depend on a particular workstation layout.

## Baselines

- `baselines/crsdiff`: CRS-Diff wrapper for remote-sensing controllable image
  generation.
- `baselines/earthsynth`: EarthSynth ControlNet wrapper for remote-sensing
  semantic-mask conditioned image generation.
- `baselines/seg2any`: Seg2Any wrapper for open-set segmentation-mask to image
  generation on the same LoveDA mask-to-RGB manifest.
- `baselines/instancediffusion`: InstanceDiffusion wrapper for open-set
  box-conditioned image generation from LoveDA semantic masks.
- `baselines/dreamcd`: DreamCD wrapper for closed-set SECOND change image
  synthesis with paired semantic masks and binary change masks.
- `baselines/place`: PLACE wrapper for semantic-mask plus class-text image
  synthesis from LoveDA masks.
- `baselines/rsedit`: RSEdit public UNet checkpoint wrapper for source-image plus
  text editing on SECOND (Table 4 candidate).
- `baselines/changebridge`: ChangeBridge SECOND data/config/train/sample
  preparation (Table 4 candidate; training required and public `runners/` is
  currently missing upstream).
- `baselines/todsynth`: TODSynth training and full-dataset TODSynth/CRFM LoveDA
  inference wrappers (Table 3 candidate; training required).
- `baselines/segearth_ov`: standalone official training-free SegEarth-OV
  inference on LoveDA, FLAIR #1, xBD-pre, and CHN6-CUG, with native-extent
  metrics and restartable saved predictions.

Run CRS-Diff on LoveDA generation using the same `cond_mask` and `gt_rgb`
folders saved by Vistar LoveDA gen eval:

```bash
bash scripts/bootstrap_crsdiff.sh
pip install -r requirements.txt
python tools/check_crsdiff_deps.py
bash run_bash/crsdiff_loveda_gen.bash
```

Run EarthSynth on the same LoveDA generation manifest:

```bash
pip install -r requirements.txt
python tools/check_earthsynth_deps.py
MAX_SAMPLES=5 bash run_bash/earthsynth_loveda_gen.bash
bash run_bash/earthsynth_loveda_gen.bash
```

Run Seg2Any on the same LoveDA generation manifest:

```bash
bash scripts/bootstrap_seg2any.sh
pip install -r requirements-seg2any.txt
python tools/check_seg2any_deps.py
MAX_SAMPLES=5 bash run_bash/seg2any_loveda_gen.bash
bash run_bash/seg2any_loveda_gen.bash
```

Run InstanceDiffusion on the same LoveDA generation manifest:

```bash
bash scripts/bootstrap_instancediffusion.sh
pip install -r requirements-instancediffusion.txt
python tools/check_instancediffusion_deps.py
MAX_SAMPLES=5 bash run_bash/instancediffusion_loveda_gen.bash
bash run_bash/instancediffusion_loveda_gen.bash
```

Run DreamCD on SECOND-style paired change generation:

```bash
pip install -r requirements-dreamcd.txt
bash scripts/bootstrap_dreamcd.sh
python tools/check_dreamcd_deps.py
SECOND_ROOT=/root/data/SECOND MAX_SAMPLES=5 bash run_bash/dreamcd_second_gen.bash
SECOND_ROOT=/root/data/SECOND bash run_bash/dreamcd_second_gen.bash
```

DreamCD weights default to `/root/data/weight/dreamcd/second/` rather than the
Git repository. The folder contains `ldm.ckpt` and `vqvae.ckpt`.

DreamCD target-B AdaIN is disabled by default, so the real target image is not
used as an inference style condition. Set `WITH_ADAIN=1` only when intentionally
reproducing the official demo behavior. Its final SECOND result directory uses
the exact Vistar image-folder contract (`source_rgb`, `cond_mask`,
`cond_mask_official`, `cond_mask_ids`, `gt_rgb`, `pred_rgb`, `absdiff`, and
`prompts`); DreamCD runtime artifacts stay outside that directory.

Run PLACE on the same LoveDA generation manifest:

```bash
bash scripts/bootstrap_place.sh
pip install -r requirements-place.txt
python tools/check_place_deps.py
MAX_SAMPLES=5 bash run_bash/place_loveda_gen.bash
bash run_bash/place_loveda_gen.bash
```

Run RSEdit on an existing Vistar SECOND evaluation directory:

```bash
pip install -r requirements-rsedit.txt
bash scripts/bootstrap_rsedit.sh
python tools/check_rsedit_deps.py
VISTAR_EVAL_DIR=/path/to/vistar_second_eval MAX_SAMPLES=5 bash run_bash/rsedit_second_gen.bash
```

Prepare ChangeBridge. No official trained task checkpoint is published. The
public repository currently omits its imported `runners/` package, so the
bootstrap/check scripts deliberately stop before training until that source is
provided by the authors:

```bash
pip install -r requirements-changebridge.txt
bash scripts/bootstrap_changebridge.sh
VISTAR_EVAL_DIR=/path/to/second_train SPLIT=train bash run_bash/changebridge_second_prepare.bash
VISTAR_EVAL_DIR=/path/to/second_val SPLIT=val bash run_bash/changebridge_second_prepare.bash
VISTAR_EVAL_DIR=/path/to/second_test SPLIT=test bash run_bash/changebridge_second_prepare.bash
```

Prepare and train TODSynth, then run the full LoveDA set (the official test
scripts stop after one batch):

```bash
pip install -r requirements-todsynth.txt
bash scripts/bootstrap_todsynth.sh
VISTAR_EVAL_DIR=/path/to/loveda_eval SPLITS=train DATASET_TAG=loveda_train \
  bash run_bash/todsynth_loveda_prepare.bash
bash run_bash/todsynth_loveda_vectorize.bash
bash run_bash/todsynth_loveda_train.bash
VISTAR_EVAL_DIR=/path/to/loveda_eval SPLITS=val DATASET_TAG=loveda_val \
  bash run_bash/todsynth_loveda_prepare.bash
DATASET_TAG=loveda_val bash run_bash/todsynth_loveda_vectorize.bash
DATASET_TAG=loveda_val TODSYNTH_CHECKPOINT=/path/to/checkpoint bash run_bash/todsynth_loveda_gen.bash
DATASET_TAG=loveda_val MMSeg_CONFIG=/path/to/config.py MMSeg_CKPT=/path/to/model.pth \
  TODSYNTH_CHECKPOINT=/path/to/checkpoint bash run_bash/todsynth_loveda_crfm_gen.bash
```
