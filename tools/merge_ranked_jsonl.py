from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge rank-sharded JSONL metadata after data-parallel inference.")
    parser.add_argument("--directory", required=True)
    parser.add_argument("--prefix", default="prompts_rank")
    parser.add_argument("--output", default="prompts.jsonl")
    args = parser.parse_args()

    directory = Path(args.directory).expanduser().resolve()
    inputs = sorted(directory.glob(f"{args.prefix}*.jsonl"))
    if not inputs:
        raise FileNotFoundError(f"No {args.prefix}*.jsonl files under {directory}")
    rows = [json.loads(line) for path in inputs for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: str(row.get("name", "")))
    output = directory / args.output
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[merge_ranked_jsonl] merged {len(rows)} rows from {len(inputs)} rank files -> {output}")


if __name__ == "__main__":
    main()
