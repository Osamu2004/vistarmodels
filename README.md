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
- `baselines/seg2any`: Seg2Any wrapper for open-set segmentation-mask to image
  generation on the same LoveDA mask-to-RGB manifest.
- `baselines/instancediffusion`: InstanceDiffusion wrapper for open-set
  box-conditioned image generation from LoveDA semantic masks.
- `baselines/place`: PLACE wrapper for semantic-mask plus class-text image
  synthesis from LoveDA masks.

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

Run PLACE on the same LoveDA generation manifest:

```bash
bash scripts/bootstrap_place.sh
pip install -r requirements-place.txt
python tools/check_place_deps.py
MAX_SAMPLES=5 bash run_bash/place_loveda_gen.bash
bash run_bash/place_loveda_gen.bash
```
