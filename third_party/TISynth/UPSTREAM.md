# Vendored TISynth upstream

- Upstream repository: https://github.com/dongrunmin/TISynth
- Vendored revision: `688cda0597b1550cb32bbae0469b8a0a900501c0`
- Upstream revision date checked: 2026-07-12

The source tree is committed directly into `vistarmodels`; runtime scripts do
not clone, fetch, or pull TISynth. Generated `__pycache__`, `.pyc`, `.DS_Store`,
model weights, and nested Git metadata are intentionally excluded.

No `LICENSE` file was present in the upstream revision. Verify redistribution
and publication terms with the upstream authors before distributing this
vendored source outside the research repository.

Local integration code lives outside this directory under:

- `baselines/tisynth/`
- `run_bash/tisynth_loveda_gen.bash`
- `run_bash/tisynth_loveda_train.bash`
- `tools/build_tisynth_loveda_manifest.py`
