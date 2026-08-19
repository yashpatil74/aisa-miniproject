#!/bin/bash
# Phase 2a: Mash sketch + all-vs-all distance matrix
# Produces: data/mash_dist_matrix.tsv
#
# Usage: bash scripts/01_mash_sketch.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FNA_DIR="$PROJECT_ROOT/fna_files"
DATA_DIR="$PROJECT_ROOT/data"
SKETCH_DIR="$DATA_DIR/mash_sketches"
THREADS=20

mkdir -p "$SKETCH_DIR" "$DATA_DIR"

echo "=== Phase 2a: Mash Sketch & Distance ==="
echo "FNA directory: $FNA_DIR"
echo "Threads: $THREADS"

# ── Step 1: Sketch all genomes ───────────────────────────────────────────────
# -k 21   : k-mer size (Mash default, optimized for ANI estimation)
# -s 10000: sketch size (higher = more accurate, 10K is robust)
# -o      : output prefix for combined sketch
echo ""
echo "[1/2] Sketching all .fna files..."
mash sketch -k 21 -s 10000 -p "$THREADS" -o "$SKETCH_DIR/all_genomes" "$FNA_DIR"/*.fna
echo "  Sketch complete: $SKETCH_DIR/all_genomes.msh"

# ── Step 2: All-vs-all distance matrix ───────────────────────────────────────
# -t : tabular output (square matrix)
# -p : parallel threads
echo ""
echo "[2/2] Computing pairwise distance matrix..."
mash dist "$SKETCH_DIR/all_genomes.msh" "$SKETCH_DIR/all_genomes.msh" \
    -p "$THREADS" -t > "$DATA_DIR/mash_dist_matrix.tsv"
echo "  Distance matrix saved: $DATA_DIR/mash_dist_matrix.tsv"

# ── Summary ──────────────────────────────────────────────────────────────────
GENOME_COUNT=$(head -1 "$DATA_DIR/mash_dist_matrix.tsv" | awk '{print NF}')
echo ""
echo "=== Done ==="
echo "  Genomes in matrix: $GENOME_COUNT"
echo "  Matrix file size: $(du -h "$DATA_DIR/mash_dist_matrix.tsv" | cut -f1)"
