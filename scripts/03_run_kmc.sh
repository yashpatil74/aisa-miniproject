#!/bin/bash
# Phase 3: KMC 31-mer extraction for all genomes
# Produces: data/kmer_dumps/<genome_id>.txt (kmer\tcount per line)
#
# Prerequisites: sudo apt install kmc
# Usage: bash scripts/03_run_kmc.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FNA_DIR="$PROJECT_ROOT/fna_files"
DUMP_DIR="$PROJECT_ROOT/data/kmer_dumps"
TMP_DIR="$PROJECT_ROOT/data/kmc_tmp"
MAX_JOBS=10

mkdir -p "$DUMP_DIR" "$TMP_DIR"

# Verify KMC is installed
if ! command -v kmc &>/dev/null; then
    echo "ERROR: kmc not found. Install with: sudo apt install kmc"
    exit 1
fi

echo "=== Phase 3: KMC 31-mer Extraction ==="
echo "FNA directory: $FNA_DIR"
echo "Output: $DUMP_DIR"
echo "Parallel jobs: $MAX_JOBS"
echo ""

# Count total files
TOTAL=$(ls "$FNA_DIR"/*.fna 2>/dev/null | wc -l)
echo "Total genomes: $TOTAL"

# Export for subshells
export DUMP_DIR TMP_DIR

# Function to process one genome
process_genome() {
    local fna_path="$1"
    local basename=$(basename "$fna_path" .fna)
    local dump_file="$DUMP_DIR/${basename}.txt"
    local tmp_subdir="$TMP_DIR/$$_${basename}"

    # Skip if already processed
    if [ -f "$dump_file" ] && [ -s "$dump_file" ]; then
        return 0
    fi

    # Create per-job tmp directory
    mkdir -p "$tmp_subdir"

    # Run KMC:
    #   -k31     : 31-mer
    #   -fm      : input is multi-FASTA
    #   -ci2     : min count 2 (drop singletons / sequencing errors)
    #   -cs65535 : max count stored (uint16)
    #   -t1      : 1 thread per job (parallelism handled by xargs)
    kmc -k31 -fm -ci2 -cs65535 -t1 \
        "$fna_path" \
        "$tmp_subdir/kmer_db" \
        "$tmp_subdir" \
        > /dev/null 2>&1

    # Dump to text
    kmc_tools transform "$tmp_subdir/kmer_db" dump "$dump_file" \
        > /dev/null 2>&1

    # Cleanup KMC database files
    rm -rf "$tmp_subdir"
}
export -f process_genome

# Run in parallel
echo ""
echo "Processing genomes (${MAX_JOBS} parallel workers)..."
ls "$FNA_DIR"/*.fna | \
    xargs -P "$MAX_JOBS" -I {} bash -c 'process_genome "$@"' _ {}

# Verify output
PROCESSED=$(ls "$DUMP_DIR"/*.txt 2>/dev/null | wc -l)
echo ""
echo "=== Done ==="
echo "  Processed: $PROCESSED / $TOTAL genomes"
echo "  Output dir size: $(du -sh "$DUMP_DIR" | cut -f1)"

if [ "$PROCESSED" -ne "$TOTAL" ]; then
    echo "  WARNING: $((TOTAL - PROCESSED)) genomes failed — check logs"
fi
