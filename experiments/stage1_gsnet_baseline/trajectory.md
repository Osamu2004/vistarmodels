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

## Attempt 1 result and failure diagnosis

### Observed results

The full two-class runs completed, but both exhibit the same foreground
collapse:

- xBD-pre: the ground-truth building prevalence is 5.92%, whereas 81.63% of
  all pixels are predicted as building. Building recall is 99.97%, precision
  is 7.26%, and building IoU is 7.26%.
- CHN6-CUG: the ground-truth road prevalence is 5.73%, whereas 46.00% of all
  pixels are predicted as road. Road recall is 98.15%, precision is 12.23%,
  and road IoU is 12.20%.

The xBD target contains exactly the same 57,962,025 positive pixels under the
RSKT-Seg run, so the WKT rasterization and metric accumulation are not the
source of the discrepancy.

### Root cause

The initial binary-vocabulary hypothesis is invalid for the released GSNet
training protocol. The official LandDiscover50K registration includes the
string `background` in its class list but sets `ignore_label=0`. The official
training forward pass excludes all pixels with that label before constructing
the BCE target. Consequently, the background output channel has no positive
background supervision; it is not a valid learned complement for a binary
`background` versus target-class argmax.

The adapter nevertheless sets `TEST_CLASS_JSON` to
`[background, road]` or `[background, building]` and converts the two sigmoid
class maps to a hard mask with `argmax`. The target class therefore wins over
an invalid background comparator across most of each image, producing the
near-one recall and very low precision observed on both datasets.

This is an evaluation-adapter error rather than ordinary cross-domain
degradation. The reported 12.20 CHN6-CUG road IoU and 7.26 xBD-pre building
IoU must be treated as invalid/provisional and must not be used as GSNet
baseline results.

### Additional protocol differences

The external native 512-tile implementation is internally consistent and the
xBD labels match the RSKT-Seg protocol. It does differ from the official GSNet
evaluation path, which enables its built-in 384-window overlapping inference
and global 640 view. This can affect accuracy and should be controlled in a
follow-up ablation, but it does not explain the shared foreground-collapse
signature.

### Required correction

Before another full run, replace the invalid background-vs-target argmax with
a foreground-compatible protocol. The two defensible candidates are:

1. emit only the target sigmoid map and apply a pre-specified one-vs-rest
   threshold; or
2. evaluate a complete foreground taxonomy and collapse the target category
   (`road` or `buildings`) against all other predicted semantic categories.

The second option is closer to GSNet's published full-taxonomy argmax
evaluation and avoids using its ignored background channel. A smoke test must
first reproduce at least one official GSNet benchmark with the released
checkpoint, then compare target-score histograms and saved masks on two xBD
and two CHN6 samples before starting full evaluation.

## Attempt 2: complete foreground taxonomy with target-class collapse

### Purpose

Remove the invalid `background` comparator while retaining the user's native
512-tile evaluation protocol. This attempt changes only the class-decision
protocol identified by Attempt 1; checkpoint, architecture, prompt template,
precision, tile size, model input size, and target masks remain unchanged.

### Implementation

- Replaced each two-entry test vocabulary with 39 non-background semantic
  classes derived from the LandDiscover50K taxonomy.
- CHN6-CUG uses `road` as model foreground. xBD-pre uses the target-dataset
  spelling `building`, consistent with GSNet's official Potsdam test
  vocabulary.
- GSNet predicts all 39 class maps, takes their per-pixel argmax, and then
  converts only the configured target-class ID to binary foreground. Every
  other semantic prediction becomes binary background.
- The evaluator rejects any test taxonomy that contains `background`, has
  duplicate or empty labels, or omits the configured target class.
- Added `pred_class_id/` outputs and aggregate/per-image predicted-class
  histograms so foreground collapse and competing categories are directly
  auditable.
- Added ground-truth and predicted foreground pixel counts/fractions to
  `metrics.json`.
- Strengthened resume validation to bind cached predictions to the prediction
  protocol, exact test-class list, target class, checkpoint, base config, CLIP
  and RSIB paths, prompt ensemble, GSNet layer count, pooling sizes, precision,
  tile size, and model input size. Historical two-class caches are rejected.
- Changed default output roots to include `fullvocab`, preventing accidental
  collision with invalid Attempt-1 outputs.
- Extended `tools/check_gsnet_deps.py` so launcher preflight validates the
  selected vocabulary and target class.

### Verification

- Python byte-compilation passed for the evaluator and dependency checker.
- Bash syntax validation passed for both launchers.
- Both JSON files parse as 39 unique non-background classes; `road` resolves
  to index 3 and `building` to index 7.
- A pure regression test verified multiclass-ID to target-vs-rest collapse and
  verified that an Attempt-1 run configuration is rejected because it lacks
  the corrected prediction protocol.
- One-GPU xBD-pre and CHN6-CUG dry runs expand to the expected 39-class
  evaluator arguments and new `fullvocab` output roots.
- Targeted `git diff --check` passes.
- No CUDA runtime is available in the local macOS workspace. Corrected metrics
  remain TODO and no paper table value is restored yet.

### Next gate

Run `MAX_SAMPLES=2 SAVE_IMAGES=1` into new smoke-test directories. Inspect the
binary masks, `pred_class_id` maps, `predicted_foreground_fraction`, and
`predicted_class_pixels`. Do not start the 903/933-image runs unless the
foreground fractions cease exhibiting Attempt 1's collapse. Reproducing an
official GSNet benchmark remains the preferred independent checkpoint/model
sanity check.
