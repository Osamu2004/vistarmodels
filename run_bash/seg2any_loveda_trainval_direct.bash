#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_URL="${BASE_URL:-http://10.200.137.14:5244}"
REMOTE_ROOT="${REMOTE_ROOT:-/baidu/vistar/results/inference}"
DATA_ROOT="${DATA_ROOT:-/data/vistar/datasets/atomic/loveda}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/vistar/runs/seg2any_loveda_trainval_direct}"
MANIFEST="${MANIFEST:-${OUTPUT_DIR}/manifest_loveda_trainval_direct.jsonl}"
ZIP_PATH="${ZIP_PATH:-/data/vistar/runs/artifacts/seg2any_loveda_trainval_direct.zip}"
SEED_FROM_OUTPUT_DIR="${SEED_FROM_OUTPUT_DIR:-/data/vistar/runs/seg2any_loveda_val_direct}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_HOME="${HF_HOME:-/data/vistar/weights/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/vistar/weights/hf_cache/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/data/vistar/weights/hf_cache/hub}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
SEG2ANY_ROOT="${SEG2ANY_ROOT:-${ROOT_DIR}/third_party/Seg2Any}"
SEG2ANY_FLUX1_MODEL="${SEG2ANY_FLUX1_MODEL:-/data/vistar/weights/flux1/FLUX.1-dev}"
SEG2ANY_LORA_CKPT="${SEG2ANY_LORA_CKPT:-/data/vistar/weights/seg2any/sacap_1m/seg2any/checkpoint-20000}"
SPLITS="${SPLITS:-train,val}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OVERWRITE="${OVERWRITE:-0}"
UPLOAD="${UPLOAD:-1}"
DRY_RUN="${DRY_RUN:-0}"

upload_file() {
  local local_path="$1"
  local remote_path="$2"
  local remote_dir="${remote_path%/*}"
  python3 - "$BASE_URL" "$remote_dir" <<'PY'
import json
import sys
import urllib.request

base, path = sys.argv[1], sys.argv[2]
data = json.dumps({"path": path}).encode()
req = urllib.request.Request(
    base.rstrip() + "/api/fs/mkdir",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST",
)
print(urllib.request.urlopen(req, timeout=30).read().decode())
PY
  curl --fail-with-body -sS --connect-timeout 20 --max-time 0 \
    -T "$local_path" \
    -H "File-Path: $remote_path" \
    -H "Content-Type: application/octet-stream" \
    "$BASE_URL/api/fs/put"
}

mkdir -p "$OUTPUT_DIR" "$(dirname "$ZIP_PATH")"

if [[ -d "$SEED_FROM_OUTPUT_DIR/pred_rgb" ]]; then
  echo "[$(date '+%F %T')] seeding existing outputs from ${SEED_FROM_OUTPUT_DIR}"
  for subdir in pred_rgb pred_rgb_native cond_mask gt_rgb seg2any_inputs; do
    if [[ -d "$SEED_FROM_OUTPUT_DIR/$subdir" ]]; then
      mkdir -p "$OUTPUT_DIR/$subdir"
      cp -an "$SEED_FROM_OUTPUT_DIR/$subdir/." "$OUTPUT_DIR/$subdir/"
    fi
  done
fi

echo "[$(date '+%F %T')] building direct LoveDA manifest splits=${SPLITS}: ${MANIFEST}"
python3 - "$DATA_ROOT" "$MANIFEST" "$SPLITS" <<'PY'
import json
import sys
from pathlib import Path

data_root = Path(sys.argv[1]).resolve()
manifest = Path(sys.argv[2]).resolve()
splits = {item.strip().lower() for item in sys.argv[3].split(",") if item.strip()}
prompt = (
    "A high-resolution remote sensing satellite image with buildings, roads, "
    "water, barren land, forest, and agriculture."
)
rows = []
seen = set()
for json_path in sorted((data_root / "json").glob("*.json")):
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    for sample in payload.get("samples", []):
        split = str(sample.get("split", "")).lower()
        if split not in splits:
            continue
        images = sample.get("images", {})
        anno = sample.get("annotation", {})
        cond_rel = anno.get("pre_class")
        target_rel = images.get("pre_image")
        if not cond_rel or not target_rel:
            continue
        cond = data_root / cond_rel
        target = data_root / target_rel
        if not cond.is_file() or not target.is_file():
            continue
        name = str(sample.get("sample_id") or cond.stem)
        if name in seen:
            continue
        seen.add(name)
        rows.append(
            {
                "name": name,
                "condition_image": str(cond),
                "target_image": str(target),
                "prompt": prompt,
            }
        )

manifest.parent.mkdir(parents=True, exist_ok=True)
with manifest.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"wrote {len(rows)} rows to {manifest}")
if not rows:
    raise SystemExit("no LoveDA rows found")
PY

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "yes" ]]; then
  echo "[$(date '+%F %T')] dry run complete"
  exit 0
fi

cd "$ROOT_DIR"
echo "[$(date '+%F %T')] start Seg2Any LoveDA ${SPLITS} on GPU ${CUDA_VISIBLE_DEVICES}"
EXTRA_ARGS=()
if [[ "$MAX_SAMPLES" != "0" ]]; then
  EXTRA_ARGS+=(--max_samples "$MAX_SAMPLES")
fi
if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" || "$OVERWRITE" == "yes" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

"$PYTHON_BIN" baselines/seg2any/run_seg2any_manifest.py \
  --seg2any_root "$SEG2ANY_ROOT" \
  --manifest "$MANIFEST" \
  --output_dir "$OUTPUT_DIR" \
  --pretrained_model_name_or_path "$SEG2ANY_FLUX1_MODEL" \
  --lora_ckpt_path "$SEG2ANY_LORA_CKPT" \
  --resolution 256 \
  --eval_size 256 \
  --batch_size 1 \
  --cond_scale_factor 2 \
  --num_inference_steps 32 \
  --guidance_scale 3.5 \
  --cond2image_attention_weight 1.0 \
  --attention_mask_method hard \
  --hard_attn_block_start 19 \
  --hard_attn_block_end 37 \
  --dtype bf16 \
  --seed 0 \
  "${EXTRA_ARGS[@]}"

echo "[$(date '+%F %T')] zipping ${OUTPUT_DIR} -> ${ZIP_PATH}"
rm -f "$ZIP_PATH"
cd "$(dirname "$OUTPUT_DIR")"
zip -qr "$ZIP_PATH" "$(basename "$OUTPUT_DIR")"

if [[ "$UPLOAD" == "1" || "$UPLOAD" == "true" || "$UPLOAD" == "yes" ]]; then
  echo "[$(date '+%F %T')] uploading ${ZIP_PATH}"
  upload_file "$ZIP_PATH" "$REMOTE_ROOT/$(basename "$ZIP_PATH")"
fi
echo "[$(date '+%F %T')] done"
