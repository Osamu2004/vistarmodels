# DynamicEarth on LEVIR-CD

This adapter runs the official DynamicEarth M-C-I instantiation used for
open-vocabulary building change detection:

1. SAM ViT-H proposes class-agnostic masks from both temporal images.
2. DINO ViT-B/16 compares the two temporal features inside each shared mask.
3. SegEarth-OV (CLIP ViT-B/16 + SimFeatUp) retains building changes.

It preserves the official LEVIR-CD script settings, including full native
1024x1024 processing, SAM proposal thresholds, and the 145-degree DINO change
threshold. This is the paper's M-C-I `SAM-DINO-SegEarth-OV` variant, not the
I-M-C APE variant.

## Setup and inference

Use a separate CUDA environment because DynamicEarth's pinned OpenMMLab and
SimFeatUp dependencies are substantially heavier than RCDNet's:

```bash
python -m pip install -r requirements-dynamicearth.txt
bash scripts/bootstrap_dynamicearth.sh
python tools/check_dynamicearth_deps.py

MAX_SAMPLES=2 bash run_bash/dynamicearth_levircd.bash
bash run_bash/dynamicearth_levircd.bash
```

The bootstrap pins official source revision
`c9ffd90cafbd791cd75a48a5717a902966c2436c`, downloads SAM ViT-H and the
official SegEarth-OV `xclip_jbu_one_million_aid.ckpt`, and builds the bundled
SAM/SimFeatUp packages. DINO and CLIP weights are downloaded by their official
loaders on first inference.

Set `FEATURE_UP=0` only as a dependency/debugging fallback. It is not the
paper-faithful M-C-I configuration and should not be used for final Figure 5.

## Outputs

The adapter writes the same standardized output tree as RCDNet:

```text
output_dir/
├── input_A/
├── input_B/
├── gt_mask/
├── pred_mask/       # <stem>_pred_mask.png, VISTAR metric-script compatible
├── pred_rgb/
├── error_map/       # TP white, TN black, FP green, FN red
├── overlay_A/
├── overlay_B/
├── metrics.json
└── per_image_metrics.jsonl
```

These saved predictions can be compared sample-for-sample with RCDNet and
VISTAR. The aggregate metrics use the exact global-pixel binary equations in
the existing VISTAR LEVIR-CD/OpenDPR evaluator.
