#!/usr/bin/env bash
# Package this repo's src/ as a Kaggle Dataset named "arc-solver-src" so the
# submission notebook can import arc_solver with no internet.
#
# This script PREPARES the upload folder and prints the exact commands — it does
# NOT upload anything. Uploading is your call (it publishes to your account).
#
# Usage:
#   scripts/prepare_kaggle_dataset.sh [KAGGLE_USERNAME]
#
# If KAGGLE_USERNAME is omitted, it is read from ~/.kaggle/kaggle.json.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/build/kaggle-dataset/arc-solver-src"

# --- Resolve the Kaggle username (needed for the dataset id "user/slug"). ---
USERNAME="${1:-}"
if [ -z "$USERNAME" ] && [ -f "$HOME/.kaggle/kaggle.json" ]; then
  USERNAME="$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["username"])' 2>/dev/null || true)"
fi
if [ -z "$USERNAME" ]; then
  echo "ERROR: pass your Kaggle username as the first argument, e.g." >&2
  echo "  scripts/prepare_kaggle_dataset.sh your_kaggle_username" >&2
  exit 1
fi

# --- Stage src/ + the dataset metadata Kaggle needs. ---
rm -rf "$OUT"
mkdir -p "$OUT"
# Copy the package but skip caches/compiled files so the dataset stays small.
( cd "$ROOT" && find src -name '__pycache__' -prune -o -type f -name '*.py' -print \
    | while read -r f; do mkdir -p "$OUT/$(dirname "$f")" && cp "$f" "$OUT/$f"; done )

cat > "$OUT/dataset-metadata.json" <<JSON
{
  "title": "arc-solver-src",
  "id": "$USERNAME/arc-solver-src",
  "licenses": [{ "name": "MIT" }]
}
JSON

echo "Prepared Kaggle dataset folder:"
echo "  $OUT"
echo "  (contains src/arc_solver/... + dataset-metadata.json)"
echo
echo "Next — upload it (requires the Kaggle CLI + an API token in ~/.kaggle/kaggle.json):"
echo "  pip install kaggle"
echo "  # first time (creates the dataset):"
echo "  kaggle datasets create   -p \"$OUT\" --dir-mode zip"
echo "  # later updates (new version):"
echo "  kaggle datasets version  -p \"$OUT\" -m \"update src\" --dir-mode zip"
echo
echo "Or upload via the web UI: kaggle.com/datasets -> New Dataset ->"
echo "drag the src/ folder in and name it exactly 'arc-solver-src'."
echo
echo "Then in the notebook: Add Data -> your 'arc-solver-src' dataset. It mounts"
echo "at /kaggle/input/arc-solver-src/src, which the setup cell finds automatically."
