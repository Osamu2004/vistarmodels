# GSNet cross-domain segmentation

This adapter evaluates the official AAAI 2025 GSNet release on four Vistar
segmentation protocols already used by the RSKT-Seg/SegEarth-OV adapters:

- **CHN6-CUG val:** road versus background.
- **xBD-pre test:** building versus background, using the same rounded
  `features.xy` WKT rasterization as the RSKT-Seg/SegEarth-OV-compatible
  evaluation.
- **FLAIR #1 test:** the official 12-class label subset.
- **UAVid:** all eight Vistar classes over the complete 270-image population.

The official `GSNet_base.pth` checkpoint was trained on LandDiscover50K.
All four evaluations are therefore cross-dataset/out-of-domain evaluations.
The public GSNet release has no UAVid configuration or UAVid checkpoint; the
UAVid entry is an explicit Vistar adaptation rather than an official GSNet
benchmark reproduction.

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

For UAVid, GSNet receives exactly the Vistar eight-class vocabulary in the
order background clutter, building, road, tree, low vegetation, moving car,
static car, and human. Background clutter is an evaluated semantic class,
not an auxiliary negative channel. No extra background, unknown, other, or
support class is appended.

UAVid frames are evaluated at native extent. Each frame is right/bottom zero
padded to a multiple of 512, divided into non-overlapping 512-pixel tiles,
predicted with eight-class argmax, stitched, and cropped back to the original
shape. GSNet retains its internal 384-pixel CLIP/RSIB encoder resolution.
Region metrics and 3-pixel WFm are computed once on each reassembled native
prediction, not independently per tile.

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

FLAIR #1 test on physical GPU 0:

```bash
GPU_IDS=0 \
NPROC_PER_NODE=1 \
DATA_ROOT='/root/data/FLAIR-1-2/data/flair#1-test' \
OUTPUT_DIR=/root/data/experiment/gsnet_flair1_test_ld50k_official640_gpu0 \
bash run_bash/gsnet_flair.bash
```

UAVid on physical GPU 1:

```bash
GPU_IDS=1 \
NPROC_PER_NODE=1 \
DATA_ROOT=/root/data/OVSISBenchDataset/uavid \
BOOTSTRAP_GSNET=0 \
CHECK_DEPS=1 \
OUTPUT_DIR=/root/data/experiment/gsnet_uavid_ld50k_vistar8_tile512_resize512_gpu1 \
bash run_bash/gsnet_uavid.bash
```

Strict UAVid evaluation requires exactly 270 paired images and labels,
`TILE_SIZE=512`, `INPUT_SIZE=512`, `NUM_LAYERS=2`,
`PROMPT_ENSEMBLE=single`, and `AMP=fp32`. Use
`MAX_SAMPLES=2 STRICT_PROTOCOL=0` with a separate output directory for a
smoke test. The primary `miou` includes all eight classes; the output also
records diagnostic `miou_foreground7` without background clutter.

The FLAIR adapter reads the complete official 15,700-patch test split
directly from its five-band GeoTIFF release and uses bands 1--3 as RGB. It
matches GSNet's released FLAIR inference path: each native 512 x 512 patch is
resized to 640 x 640, evaluated with overlapping 384 x 384 local windows and
a 384 x 384 global view, and restored to 512 x 512 for metrics. Raw labels
1--12 form the evaluated classes; raw 0, 13--19, and 255 are ignored. The
script reports mIoU, mAcc, mF1, pixel accuracy, and the 3-pixel boundary WFm,
and saves class-ID and common-palette RGB prediction/ground-truth masks.

This default run uses the public LandDiscover50K `GSNet_base.pth`. It is not
the OVRSISBenchV2 ViT-B model retrained on OVRSIS95K. Do not combine its WFm
with the published OVRSISBenchV2 FLAIR mIoU in one table row; both metrics
must come from the same checkpoint and preprocessing.

Use `NPROC_PER_NODE=2 GPU_IDS=0,1` for independent two-GPU data parallel
evaluation. Use `MAX_SAMPLES=2` for a smoke test and `DRY_RUN=1` to print the
resolved command without loading the model. Do not reuse the historical
two-class output directories. The resume guard rejects cached predictions
whose taxonomy, target class, checkpoint, prompt ensemble, layer count, or
prediction protocol differs.
