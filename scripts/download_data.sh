#!/usr/bin/env bash
# Download training data: Lichess Elite PGNs + puzzle CSV.
# Run once per fresh pod.
set -euo pipefail
DATA_DIR="${1:-data/raw}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "--- Lichess puzzle DB (compressed ~280MB) ---"
if [[ ! -f lichess_db_puzzle.csv ]]; then
  wget -q --show-progress https://database.lichess.org/lichess_db_puzzle.csv.zst
  zstd -d lichess_db_puzzle.csv.zst
fi

echo "--- Lichess Elite DB (select months) ---"
# The Elite DB is split by month. Grab the last 6 months for ~500k games.
BASE="https://database.nikonoel.fr/lichess_elite"
for m in 2025-10 2025-09 2025-08 2025-07 2025-06 2025-05; do
  f="lichess_elite_${m}.pgn.zst"
  if [[ ! -f "lichess_elite_${m}.pgn" ]]; then
    wget -q --show-progress "${BASE}_${m}.pgn.zst" -O "$f" || echo "(skipping ${m}: not published yet)"
    [[ -f "$f" ]] && zstd -d "$f" && rm -f "$f"
  fi
done

echo "Done. Files:"
ls -lh .
