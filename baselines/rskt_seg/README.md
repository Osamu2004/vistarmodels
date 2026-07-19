# RSKT-Seg on CHN6-CUG

This adapter evaluates the official
[RSKT-Seg](https://github.com/LiBingyu01/RSKT-Seg) DLRSD-trained ViT-L
checkpoint on CHN6-CUG road segmentation. CHN6-CUG is not registered by the
official repository, so this wrapper supplies the two-class vocabulary and
computes the binary road metrics without modifying upstream source.

## Evaluation protocol

- Model: released RSKT-Seg DLRSD + CLIP ViT-L/14@336 checkpoint.
- Adaptation: none; this is cross-dataset/out-of-domain evaluation.
- Text classes: `background`, `road`.
- Ground truth: value zero is background and every nonzero value is road.
- Primary reported value: foreground `road_iou`.
- Additional outputs: road F1/precision/recall, background IoU, binary mIoU,
  and pixel accuracy.
- Default input size: 512, with metrics computed at each original image size.

The output folder follows the other segmentation baselines:
`input`, `pred_mask`, `pred_rgb`, `gt_mask`, `gt_rgb`, `overlay`,
`predictions.jsonl`, and `metrics.json`.

## Weights

RSKT-Seg has public weights, but the official main checkpoint is hosted in a
Baidu/OneDrive folder rather than a direct Hugging Face file. Download the
DLRSD + ViT-L checkpoint from the
[official weight link](https://pan.baidu.com/s/1xX6TBLAn3Xypsq-IZI3azw?pwd=USTC)
and place it at:

```text
/root/data/weight/rskt_seg/RSKT_Seg_DLRSD_ViT_L/model_final.pth
```

The public checkpoint also needs four foundation weights:

```text
/root/data/weight/rskt_seg/pretrained/ViT-L-14-336px.pt
/root/data/weight/rskt_seg/pretrained/ViT-B-32.pt
/root/data/weight/rskt_seg/pretrained/RemoteCLIP-ViT-B-32.pt
/root/data/weight/rskt_seg/pretrained/RSIB.pth
```

The launcher and bootstrap script automatically download these four auxiliary
files when they are absent. `RemoteCLIP-ViT-B-32.pt` is downloaded from its
Hugging Face repository; the two OpenAI CLIP files use their official direct
URLs, and `RSIB.pth` uses its public Google Drive file. The only manual file is
the official RSKT-Seg DLRSD + ViT-L `model_final.pth`, because the authors
currently publish it only through Baidu Netdisk and OneDrive folders and no
official Hugging Face copy is available.

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
bash run_bash/rskt_seg_chn6_road.bash
```

Use `MAX_SAMPLES=2` for a smoke test. Multi-GPU inference splits files across
independent model processes and uses Gloo only for CPU synchronization; it
does not use NCCL.
