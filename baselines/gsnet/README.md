# GSNet cross-domain target-class segmentation

This adapter evaluates the official AAAI 2025 GSNet release on two
target-class segmentation protocols already used by the RSKT-Seg adapter:

- **CHN6-CUG val:** road versus background.
- **xBD-pre test:** building versus background, using the same rounded
  `features.xy` WKT rasterization as the RSKT-Seg/SegEarth-OV-compatible
  evaluation.

The official `GSNet_base.pth` checkpoint was trained on LandDiscover50K.
Both evaluations are therefore cross-dataset/out-of-domain evaluations.

## Foreground prediction protocol

Do not evaluate the released checkpoint with a two-class
`[background, target]` vocabulary. LandDiscover50K label 0 is registered as
the ignored label in the official GSNet source. During training, it is never
encoded as a positive `background` target, so that text channel is not a
learned complement for binary argmax. The invalid two-class adapter produces
near-all-foreground masks on both xBD-pre and CHN6-CUG.

The corrected adapter supplies the 39 non-background LandDiscover50K semantic
classes at inference. It takes the argmax over that complete foreground
taxonomy and then collapses `building` (xBD-pre) or `road` (CHN6-CUG) to
binary foreground; every other predicted semantic category becomes binary
background. This is substantially closer to the official GSNet evaluation,
which performs argmax over a complete target taxonomy. The xBD vocabulary uses
the target-dataset spelling `building`, consistent with the official GSNet
Potsdam evaluation vocabulary.

## Resolution protocol

Inference uses native, non-overlapping 512 x 512 source tiles. Images are
right/bottom zero padded only when their size is not divisible by 512, tile
predictions are stitched, padding is cropped, and metrics are accumulated at
the original image size. With the default `INPUT_SIZE=512`, there is no
external tile resize. Inside GSNet, the official model resamples the CLIP and
RSIB/DINO encoder inputs to 384 x 384; this fixed internal encoder resolution
does not change the 512-tile evaluation protocol.

The output tree matches the RSKT-Seg adapters:

```text
OUTPUT_DIR/
  input/
  pred_mask/
  pred_class_id/
  pred_rgb/
  gt_mask/
  gt_rgb/
  overlay/
  class_map.json
  run_config.json
  predictions.jsonl
  metrics.json
```

## Setup

```bash
cd /root/code/vistarmodels
python -m pip install -r requirements-gsnet.txt
bash scripts/bootstrap_gsnet.sh
python tools/check_gsnet_deps.py
```

The bootstrap pins the official source revision, downloads the official
`GSNet_base.pth`, downloads OpenAI CLIP ViT-B/16, and reuses
`/root/data/weight/rsib/RSIB.pth` when present. Set `GSNET_DOWNLOAD_WEIGHTS=0`
to disable weight downloads. If Detectron2 is not already installed, first
bootstrap RSKT-Seg and install its bundled Detectron2 source, or set
`GSNET_INSTALL_DETECTRON2=1`.

This RSIB reuse is intentional. The official GSNet and RSKT-Seg repositories
link the same Google Drive file ID, use identical `BuildRSIB` loaders, and
carry byte-identical `vision_transformer.py` implementations. Only the RSIB
specialist encoder is shared: GSNet still uses its own LandDiscover50K model
checkpoint and CLIP ViT-B/16 branch, whereas the selected RSKT-Seg release
uses its own DLRSD model checkpoint, CLIP ViT-L/14@336 branch, and RemoteCLIP
branch.

Official sources:

- Code: <https://github.com/yecy749/GSNet>
- GSNet checkpoint:
  <https://drive.google.com/file/d/1YMAZj5fMUI3uSCvUmGHzyf4LthXdji0/view>
- RSIB checkpoint:
  <https://drive.google.com/file/d/1kH0wDM_Hl4sEQJG8JjILCo0RTx65X7zV/view>

## Run

CHN6-CUG on physical GPU 0:

```bash
GPU_IDS=0 \
NPROC_PER_NODE=1 \
DATA_ROOT=/root/data/CHN6-CUG/val \
OUTPUT_DIR=/root/data/experiment/gsnet_chn6_cug_ld50k_fullvocab_gpu0_tile512 \
bash run_bash/gsnet_chn6_road.bash
```

xBD-pre on physical GPU 0:

```bash
GPU_IDS=0 \
NPROC_PER_NODE=1 \
DATA_ROOT=/root/data/xview2/test \
OUTPUT_DIR=/root/data/experiment/gsnet_xbd_pre_ld50k_fullvocab_gpu0_tile512 \
bash run_bash/gsnet_xbd_pre_building.bash
```

Use `NPROC_PER_NODE=2 GPU_IDS=0,1` for independent two-GPU data parallel
evaluation. Use `MAX_SAMPLES=2` for a smoke test and `DRY_RUN=1` to print the
resolved command without loading the model. Do not reuse the historical
two-class output directories. The resume guard rejects cached predictions
whose taxonomy, target class, checkpoint, prompt ensemble, layer count, or
prediction protocol differs.
