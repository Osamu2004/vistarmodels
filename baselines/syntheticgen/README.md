# SyntheticGen

This adapter evaluates the authors' released LoveDA model on the exact Table 3
validation masks. It keeps the supplied semantic mask fixed
(`strength_layout=0`) and runs only the published ratio-conditioned ControlNet
renderer. The class ratios and urban/rural prompt are derived from each input
mask. Set `SYNTHETICGEN_WEIGHT_DIR` to the local directory containing the
published layout and ControlNet checkpoints.

```bash
bash scripts/bootstrap_syntheticgen.sh
VISTAR_EVAL_DIR=/data/vistar/runs/seg2any_loveda_val_512_steps32_cfg3p5_seed0 \
  MAX_SAMPLES=5 bash run_bash/syntheticgen_loveda_gen.bash
```

The wrapper writes the common `cond_mask`, `gt_rgb`, `pred_rgb`, `absdiff`, and
`prompts` folders. It does not run SyntheticGen's Stage-A layout generator,
because Table 3 supplies the conditioning LoveDA mask.
