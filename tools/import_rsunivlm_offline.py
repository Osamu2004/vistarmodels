"""Fast-forward a local bundle while preserving the user's dataset-path JSON.

No reset, stash, staging, force merge, model loading, or data-directory edits.
Other local modifications retain Git's usual overwrite protection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--bundle", required=True)
    args = parser.parse_args()
    repo, bundle = Path(args.repo).resolve(), Path(args.bundle).resolve()
    subprocess.run(["git", "bundle", "verify", str(bundle)], cwd=repo, check=True)
    relative = "configs/rsunivlm/paper_data_roots.json"
    path = repo / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected a regular existing path configuration: {path}")
    original = path.read_bytes()
    if not isinstance(json.loads(original), dict):
        raise ValueError("Dataset-path configuration must be a JSON object")
    committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=repo)
    backup = Path(tempfile.mkdtemp(prefix="rsunivlm-paths-backup-")) / path.name
    backup.write_bytes(original)
    print(f"Dataset-path backup: {backup}", flush=True)
    try:
        # Temporarily remove only this known configuration's unstaged edit.
        # Always restore its exact original bytes, even if Git rejects other
        # local changes, staged edits, or a divergent branch.
        path.write_bytes(committed)
        subprocess.run(["git", "pull", "--ff-only", str(bundle), "main"], cwd=repo, check=True)
    finally:
        path.write_bytes(original)
        print(f"Original dataset paths restored: {path}", flush=True)
    subprocess.run(["git", "log", "-1", "--oneline"], cwd=repo, check=True)


if __name__ == "__main__":
    main()
