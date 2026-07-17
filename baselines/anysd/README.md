# AnySD

This wrapper runs the official [AnySD](https://github.com/weichow23/AnySD) visual-segmentation expert as a zero-shot SECOND change-generation baseline.

## Input contract

By default, each directional SECOND record gives the model:

1. the real source-time image;
2. the **complete target-side multi-class semantic change mask**, rendered with the fixed SECOND palette;
3. a direction-aware editing instruction containing AnySD's official `[V*]` visual placeholder, the classes present in the mask, and a compact color-name map.

All changed categories are retained in a single color mask; no category is sampled or discarded. For `t1_to_t2`, the source is T1 and the mask uses target-side T2 classes. For `t2_to_t1`, the source is T2 and the mask uses target-side T1 classes. The target-time image is never passed to the model.

The fixed prompt vocabulary is `changed inland water`, `changed bare land`, `changed grass`, `changed forest`, `changed building`, and `changed playground`. The prompt uses compact color names so it stays within CLIP's 77-token context; exact RGB triplets are saved in `class_map.json`. The saved `cond_mask_ids`, `cond_mask`, and `cond_mask_official` files retain the same evaluation-folder contract as the other SECOND generation baselines.

Set `MASK_MODE=oneclass` to reproduce the former shared random-class protocol. Only this compatibility mode requires `CLASS_SELECTION_FILE`.

## Model size

- active parameters: approximately 3.0B for the visual expert path;
- downloaded AnySD subset plus SD 1.5 base: approximately 13--14 GB;
- recommended inference memory: 16--24 GB per process at FP16.

## Run

```bash
cd /root/code/vistarmodels
python -m pip install -r requirements-anysd.txt

GPU_IDS=0,1 \
NPROC_PER_NODE=2 \
SECOND_ROOT=/root/data/second_dataset \
COMPUTE_METRICS=1 \
bash run_bash/anysd_second_gen.bash
```

Use `MAX_SAMPLES=2` for a smoke test. `BOOTSTRAP_ANYSD=1` downloads the official source, the `visual_seg` expert, the AnySD UNet/image encoder, and the required SD 1.5 components automatically.

The default output directory starts with
`/root/data/experiment/anysd_second_test_both_multiclass_targetmask_visualseg_`.

## Legacy one-class compatibility

```bash
MASK_MODE=oneclass \
CLASS_SELECTION_FILE=/root/data/experiment/protocols/second_test_oneclass_targetmask_both_resize256_seed42_labelpairauto.jsonl \
bash run_bash/anysd_second_gen.bash
```
