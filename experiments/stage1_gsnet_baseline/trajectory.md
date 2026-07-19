# Stage 1 trajectory: GSNet binary open-vocabulary segmentation

## Goal

Reproduce the official GSNet inference graph with its released
LandDiscover50K-trained checkpoint, then evaluate it out of domain on
CHN6-CUG road extraction and xBD-pre building extraction under the same
native 512-tile protocol used by the repository's RSKT-Seg baseline.

## Reference

- Official repository: <https://github.com/yecy749/GSNet>
- Pinned source revision: `61da3017529a99f8ae1bad5d423e62e2c7484e36`
- Official model checkpoint: `GSNet_base.pth`, Google Drive file
  `1YMAZj5fMUI3uSCvUmGHzyf4LthXdji0Y`
- Official RSIB checkpoint: Google Drive file
  `1kH0wDM_Hl4sEQJG8JjILCo0RTx65X7zV`
- Official model configuration: `configs/vitb_384.yaml`
- Training source: LandDiscover50K, 40 classes

No reusable experiment-memory file was present at
`/memory/experiment-memory.md` when this baseline was added.

## Attempt 1: official model with unified 512 evaluation

### Hypothesis

The official GSNet checkpoint can be evaluated on unseen binary label spaces
by replacing only `TEST_CLASS_JSON` with `[background, road]` or
`[background, building]`, while retaining its official CLIP ViT-B/16,
RSIB/DINO, fusion architecture, checkpoint loader, and single-prompt
inference.

### Implementation

- Added one generic evaluator for CHN6-CUG and xBD-pre.
- Kept source images at native resolution, used non-overlapping 512 x 512
  tiles, padded only the right/bottom boundary, stitched tile predictions,
  cropped padding, and computed metrics at original resolution.
- Reused the RSKT-Seg xBD WKT rasterizer to prevent label-protocol drift.
- Preserved GSNet's internal 384 x 384 CLIP and RSIB/DINO encoder inputs.
- Used exact binary vocabularies without prompt tuning.
- Added official-source bootstrap, official-weight downloads, dependency
  checks, resumable outputs, one- and multi-GPU launchers, and dry-run mode.

### Fixed configuration

- Checkpoint: official `GSNet_base.pth`
- Training data: LandDiscover50K
- External tile size: 512
- Model input size: 512
- Internal CLIP/RSIB encoder size: 384
- Prompt ensemble: `single`
- GSNet layers: 2
- Pooling sizes: `[1, 1]`, following the official evaluation script
- Precision: FP32 by default
- CHN6-CUG classes: `background`, `road`
- xBD-pre classes: `background`, `building`

### Verification

- Python compilation and class-JSON parsing: passed with Python 3.12.
- Bash syntax: passed for the bootstrap and both launchers.
- Dry-run command resolution: passed for one-GPU CHN6-CUG and two-GPU
  xBD-pre commands.
- `git diff --check`: passed for all GSNet files.
- Pure helper import testing was not available in the local macOS environment
  because that environment does not have OpenCV; this does not replace the
  required WSL CUDA smoke test.
- Full CUDA metrics: not run on the local macOS workspace; these require the
  user's WSL CUDA environment and datasets.

### Gate status

Stage 1 is **implementation-complete but metric-pending**. The baseline should
not advance to tuning. First run `MAX_SAMPLES=2` on each dataset, inspect the
checkpoint load report and saved masks, then run the full official splits.
