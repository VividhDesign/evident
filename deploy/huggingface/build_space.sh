#!/usr/bin/env bash
# Assembles the Hugging Face Space folder: Dockerfile + Space card + precomputed data.
# (The application code is cloned from GitHub when the Space builds its image.)
#
#   deploy/huggingface/build_space.sh [output-dir]
#   hf upload <hf-username>/evident <output-dir> . --repo-type space
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${1:-$ROOT/../evident-space}"

rm -rf "$OUT"
mkdir -p "$OUT/data/cache/embeddings" "$OUT/data/indexes"
cp "$HERE/Dockerfile" "$HERE/README.md" "$OUT/"
for d in fiqa scifact nfcorpus; do
  mkdir -p "$OUT/data/beir/$d/qrels"
  cp "$ROOT/data/beir/$d/corpus.jsonl" "$ROOT/data/beir/$d/queries.jsonl" "$OUT/data/beir/$d/"
  cp "$ROOT/data/beir/$d/qrels/test.tsv" "$OUT/data/beir/$d/qrels/"
done
# Embeddings are looked up by a hash of (model, texts), so the server finds these instead of
# re-embedding 57k documents on a small CPU.
cp "$ROOT"/data/cache/embeddings/*.npy "$OUT/data/cache/embeddings/"
cp -R "$ROOT/data/indexes/strata-docs" "$OUT/data/indexes/" 2>/dev/null || true
du -sh "$OUT"
