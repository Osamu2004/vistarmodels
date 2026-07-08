# vistarmodels

Baselines and wrappers for evaluating generation models against Vistar/UniGen.

This repository is intentionally separate from the main `vistar` codebase. It
keeps external comparison methods reproducible without mixing their dependencies
and source code into the training repository.

## Baselines

- `baselines/crsdiff`: CRS-Diff wrapper for remote-sensing controllable image
  generation.
- `baselines/earthsynth`: EarthSynth ControlNet wrapper for remote-sensing
  semantic-mask conditioned image generation.

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
