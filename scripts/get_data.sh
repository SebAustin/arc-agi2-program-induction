#!/usr/bin/env bash
# Download the public ARC-AGI-1 and ARC-AGI-2 datasets into data/.
# Both are public GitHub repositories; nothing here is competition-private.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/data"
TMP="$DATA/tmp"
mkdir -p "$TMP"

echo "Cloning ARC-AGI-1 (fchollet/ARC-AGI)..."
git clone --depth 1 -q https://github.com/fchollet/ARC-AGI.git "$TMP/arc1"
echo "Cloning ARC-AGI-2 (arcprize/ARC-AGI-2)..."
git clone --depth 1 -q https://github.com/arcprize/ARC-AGI-2.git "$TMP/arc2"

mkdir -p "$DATA"/arc_agi_1/{training,evaluation} "$DATA"/arc_agi_2/{training,evaluation}
cp "$TMP"/arc1/data/training/*.json   "$DATA/arc_agi_1/training/"
cp "$TMP"/arc1/data/evaluation/*.json "$DATA/arc_agi_1/evaluation/"
cp "$TMP"/arc2/data/training/*.json   "$DATA/arc_agi_2/training/"
cp "$TMP"/arc2/data/evaluation/*.json "$DATA/arc_agi_2/evaluation/"
rm -rf "$TMP"

echo "Done:"
echo "  arc_agi_1/training:   $(ls "$DATA"/arc_agi_1/training | wc -l | tr -d ' ')"
echo "  arc_agi_1/evaluation: $(ls "$DATA"/arc_agi_1/evaluation | wc -l | tr -d ' ')"
echo "  arc_agi_2/training:   $(ls "$DATA"/arc_agi_2/training | wc -l | tr -d ' ')"
echo "  arc_agi_2/evaluation: $(ls "$DATA"/arc_agi_2/evaluation | wc -l | tr -d ' ')"
