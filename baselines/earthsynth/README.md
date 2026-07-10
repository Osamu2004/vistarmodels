# EarthSynth Baseline

This folder contains the EarthSynth wrapper used for Vistar/UniGen mask-to-RGB
generation comparison.

EarthSynth is used through the public Diffusers interface:

- ControlNet: `jaychempan/EarthSynth`
- ControlNet subfolder: `controlnet`
- Base model: `stable-diffusion-v1-5/stable-diffusion-v1-5`

The wrapper reads the same JSONL manifest format as the CRS-Diff wrapper:

```json
{"name": "sample", "condition_image": "/path/to/mask.png", "target_image": "/path/to/rgb.png", "prompt": "A satellite image ..."}
```

It writes a Vistar-compatible output layout:

```text
output_dir/
  pred_rgb/
  pred_rgb_native/
  cond_mask/
  gt_rgb/
  manifest_resolved.jsonl
```

## Run LoveDA Mask-to-RGB

```bash
cd /root/code/vistarmodels
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=1 bash run_bash/earthsynth_loveda_gen.bash
```

Default merged train+val input:

```text
/root/data/experiment/eval_loveda_gen_gen_only_step300000
```

Default output keeps the legacy val-only directory so existing predictions can
be reused in place:

```text
/root/data/experiment/earthsynth_loveda_val_mask_to_rgb_gen_resize512_steps50_scale7p5_no_controlprompt_seed0
```

The main bash verifies that the source contains all 4,191 LoveDA train+val
samples. With the default `OVERWRITE=0`, it checks each existing
`pred_rgb/<name>_pred_rgb.png`, skips valid predictions, and generates only
missing samples.

Smoke test:

```bash
MAX_SAMPLES=5 bash run_bash/earthsynth_loveda_gen.bash
```

Resume is enabled by default. Existing files in `pred_rgb/` are skipped unless
`OVERWRITE=1` is set.

If using a local folder that directly contains the ControlNet `config.json`, run
with:

```bash
EARTHSYNTH_CONTROLNET_MODEL=/path/to/EarthSynth/controlnet \
EARTHSYNTH_CONTROLNET_SUBFOLDER= \
bash run_bash/earthsynth_loveda_gen.bash
```

## Fairness Notes

EarthSynth is a remote-sensing ControlNet model conditioned on semantic masks
and text. It is a stronger and fairer mask-to-RGB baseline than text-only remote
sensing generators, but it is still not the same task setting as UniGen because
it is not trained for bidirectional image/mask generation or change tasks.
