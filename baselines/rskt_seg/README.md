# RSKT-Seg on CHN6-CUG and xBD-pre

This adapter evaluates the official
[RSKT-Seg](https://github.com/LiBingyu01/RSKT-Seg) DLRSD-trained ViT-L
checkpoint on CHN6-CUG road segmentation and xBD-pre building extraction.
Neither target dataset is registered by the official repository, so these
wrappers supply the two-class vocabularies and compute binary metrics without
modifying upstream source.

## CHN6-CUG protocol

- Model: released RSKT-Seg DLRSD + CLIP ViT-L/14@336 checkpoint.
- Adaptation: none; this is cross-dataset/out-of-domain evaluation.
- Text classes: `background`, `road`.
- Ground truth: value zero is background and every nonzero value is road.
- Primary reported value: foreground `road_iou`.
- Additional outputs: road F1/precision/recall, background IoU, binary mIoU,
  and pixel accuracy.
- Inference: non-overlapping native-resolution 512x512 source tiles, matching
  the VISTAR CHN6-CUG protocol.
- Boundary handling: zero-pad only the right and bottom edges, stitch tile
  predictions, then crop to the exact original extent.
- Model input size: 512x512 per tile with no tile resize by default.
- Metrics: computed after stitching at each image's original spatial extent.

The output folder follows the other segmentation baselines:
`input`, `pred_mask`, `pred_rgb`, `gt_mask`, `gt_rgb`, `overlay`,
`predictions.jsonl`, and `metrics.json`.

## xBD-pre protocol

- Model: the same released RSKT-Seg DLRSD + CLIP ViT-L/14@336 checkpoint.
- Adaptation: none; this is cross-dataset/out-of-domain evaluation.
- Population: the 933 `*_pre_disaster` images in the official xBD test split.
- Text classes: `background`, `building`.
- Ground truth: rasterize `features.xy[*].wkt` polygon exteriors after rounding
  their coordinates, matching the xView2/SegEarth-OV conversion; every
  annotated polygon is building ID 1.
- Primary reported value: foreground `building_iou`; the binary mIoU that
  includes background is saved only as an auxiliary diagnostic.
- Inference: partition every native 1024x1024 image into four non-overlapping
  512x512 source tiles, predict the tiles sequentially, stitch them, and score
  at the original extent.

Raw annotation JSON files under `test/labels` are rasterized on the fly.
Already converted masks under `targets`, `targets_cvt`, or `masks_building`
are also accepted.

## Weights

RSKT-Seg has public weights, but the official main checkpoint is hosted in a
Baidu/OneDrive folder rather than a direct Hugging Face file. Download the
DLRSD + ViT-L checkpoint from the
[official weight link](https://pan.baidu.com/s/1xX6TBLAn3Xypsq-IZI3azw?pwd=USTC)
and place it at:

```text
/root/data/weight/RSKT-Seg-ckpt/0SAVEoutput_vitl_336_DLRSD_rotate_dino_remoteclip_3W_layer5/model_final.pth
```

The official folder contains multiple ViT-B/ViT-L and DLRSD/iSAID runs. The
CHN6-CUG launcher deliberately selects the ViT-L/336 DLRSD layer-5 checkpoint
above; another file can be supplied explicitly through `RSKT_CHECKPOINT`.

The selected DLRSD + ViT-L checkpoint needs three foundation weights:

```text
/root/data/weight/rskt_seg/pretrained/ViT-L-14-336px.pt
/root/data/weight/rskt_seg/pretrained/RemoteCLIP-ViT-B-32.pt
/root/data/weight/rsib/RSIB.pth
```

The launcher and bootstrap script automatically download these three auxiliary
files when they are absent. `RemoteCLIP-ViT-B-32.pt` is downloaded from its
Hugging Face repository; OpenAI CLIP ViT-L uses its official direct URL, and
`RSIB.pth` uses its public Google Drive file. An existing nonempty RSIB file at
the path above is used directly without another Google Drive download. Set
`RSKT_RSIB` to override this location. The ordinary OpenAI CLIP
ViT-B/32 weight is not used by this ViT-L configuration and is therefore not
downloaded or required. Set `RSKT_DOWNLOAD_CLIP_VITB=1` only for a custom
ViT-B configuration. The only manual file is the official RSKT-Seg DLRSD +
ViT-L `model_final.pth`, because the authors currently publish it only through
Baidu Netdisk and OneDrive folders and no official Hugging Face copy is
available.

GSNet and RSKT-Seg officially use the same RSIB/DINO specialist checkpoint:
both releases link Google Drive file ID
`1kH0wDM_Hl4sEQJG8JjILCo0RTx65X7zV`, and their `BuildRSIB` loaders are
identical. The shared path is intentional; it does not mean that their main
segmentation checkpoints or CLIP branches are shared. The bootstrap validator
loads this trusted legacy release with `weights_only=False`, which is required
under PyTorch 2.6+ and prevents a valid RSIB checkpoint from being
misclassified as corrupt and downloaded repeatedly.

## Setup

The official environment uses Python 3.8, PyTorch 2.3, CUDA 11.8, and its
bundled Detectron2 source. A separate environment is recommended.

```bash
cd /root/code/vistarmodels

python -m pip install -r requirements-rskt-seg.txt

RSKT_INSTALL_DETECTRON2=1 \
bash scripts/bootstrap_rskt_seg.sh

python tools/check_rskt_seg_deps.py
```

The bootstrap step clones the official source and downloads every
machine-accessible auxiliary weight by default. To prepare only the source,
set `RSKT_DOWNLOAD_AUX_WEIGHTS=0`.

## Run

Single GPU:

```bash
cd /root/code/vistarmodels

GPU_IDS=0 \
NPROC_PER_NODE=1 \
DATA_ROOT=/root/data/CHN6-CUG/val \
TILE_SIZE=512 \
INPUT_SIZE=512 \
bash run_bash/rskt_seg_chn6_road.bash
```

The launcher invokes the bootstrap automatically and skips files that already
exist. It stops at the dependency check with the two official manual-download
links if `model_final.pth` is still absent.

Two GPUs:

```bash
GPU_IDS=0,1 \
NPROC_PER_NODE=2 \
DATA_ROOT=/root/data/CHN6-CUG/val \
TILE_SIZE=512 \
INPUT_SIZE=512 \
bash run_bash/rskt_seg_chn6_road.bash
```

Use `MAX_SAMPLES=2` for a smoke test. Multi-GPU inference splits files across
independent model processes and uses Gloo only for CPU synchronization; it
does not use NCCL. The official RSKT-Seg non-sliding forward path emits only
the first item in `batched_inputs`, so this adapter deliberately evaluates
source tiles sequentially within each process.

xBD-pre, single GPU:

```bash
cd /root/code/vistarmodels

BOOTSTRAP_RSKT_SEG=0 \
RSKT_RSIB=/root/data/weight/rsib/RSIB.pth \
GPU_IDS=0 \
NPROC_PER_NODE=1 \
DATA_ROOT=/root/data/xview2/test \
TILE_SIZE=512 \
INPUT_SIZE=512 \
OUTPUT_DIR=/root/data/experiment/rskt_seg_xbd_pre_vitl336_dlrsd_gpu0_tile512 \
bash run_bash/rskt_seg_xbd_pre_building.bash
```

Use `MAX_SAMPLES=2` first to verify the JSON rasterization and qualitative
building masks before launching all 933 images.
