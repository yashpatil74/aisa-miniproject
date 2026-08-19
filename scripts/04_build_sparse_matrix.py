#!/usr/bin/env python3
"""
Phase 4: Build scipy sparse CSR matrix from KMC k-mer text dumps.

Two-pass approach to stay within 16 GB RAM:
  Pass 1 — Build global vocabulary (k-mer → column index), filter to k-mers in ≥2 genomes
  Pass 2 — Build COO sparse matrix using the vocabulary

Input:  data/kmer_dumps/*.txt      (kmer<tab>count per line)
        data/labels_final.csv      (for genome ordering)
Output: data/kmer_matrix.npz       (scipy CSR sparse matrix)
        data/kmer_vocab.pkl        (dict: kmer_string → column_index)
        data/genome_order.txt      (genome IDs in row order)
"""

import os
import sys
import gc
import pickle
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, save_npz
import pandas as pd
from collections import defaultdict

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMP_DIR     = os.path.join(PROJECT_ROOT, "data", "kmer_dumps")
LABELS_CSV   = os.path.join(PROJECT_ROOT, "data", "labels_final.csv")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "data")
MATRIX_PATH  = os.path.join(OUTPUT_DIR, "kmer_matrix.npz")
VOCAB_PATH   = os.path.join(OUTPUT_DIR, "kmer_vocab.pkl")
ORDER_PATH   = os.path.join(OUTPUT_DIR, "genome_order.txt")

# ── Parameters ───────────────────────────────────────────────────────────────
MIN_GENOME_FREQ = 2   # Drop k-mers appearing in fewer than this many genomes


def get_genome_list():
    """Get ordered genome list from labels file, filtering to those with dumps."""
    labels = pd.read_csv(LABELS_CSV)
    labels["genome_id"] = labels["genome_id"].astype(str)

    genome_ids = []
    for gid in labels["genome_id"]:
        dump_file = os.path.join(DUMP_DIR, f"{gid}.txt")
        if os.path.isfile(dump_file):
            genome_ids.append(gid)
        else:
            print(f"  WARNING: No dump for {gid} — skipping")

    return genome_ids


def pass1_build_vocab(genome_ids):
    """Pass 1: Count how many genomes each k-mer appears in, build filtered vocab."""
    print(f"\n{'='*60}")
    print(f"PASS 1: Building vocabulary (min_genome_freq={MIN_GENOME_FREQ})")
    print(f"{'='*60}")

    # Count genome frequency for each k-mer
    kmer_genome_count = defaultdict(int)
    total = len(genome_ids)

    for i, gid in enumerate(genome_ids):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  Scanning genome {i+1}/{total}...")

        dump_file = os.path.join(DUMP_DIR, f"{gid}.txt")
        with open(dump_file, "r") as f:
            for line in f:
                kmer = line.split("\t", 1)[0]
                kmer_genome_count[kmer] += 1

    print(f"  Total unique k-mers seen: {len(kmer_genome_count):,}")

    # Filter: keep k-mers in ≥ MIN_GENOME_FREQ genomes
    vocab = {}
    col_idx = 0
    for kmer, count in kmer_genome_count.items():
        if count >= MIN_GENOME_FREQ:
            vocab[kmer] = col_idx
            col_idx += 1

    dropped = len(kmer_genome_count) - len(vocab)
    print(f"  After filtering (≥{MIN_GENOME_FREQ} genomes): {len(vocab):,} k-mers")
    print(f"  Dropped: {dropped:,} singleton k-mers ({dropped/max(len(kmer_genome_count),1)*100:.1f}%)")

    # Free memory
    del kmer_genome_count
    gc.collect()

    return vocab


def pass2_build_matrix(genome_ids, vocab):
    """Pass 2: Build COO sparse matrix using the vocabulary."""
    print(f"\n{'='*60}")
    print(f"PASS 2: Building sparse matrix")
    print(f"{'='*60}")

    n_genomes = len(genome_ids)
    n_features = len(vocab)
    total = n_genomes

    rows = []
    cols = []
    data = []

    for row_idx, gid in enumerate(genome_ids):
        if (row_idx + 1) % 500 == 0 or row_idx == 0:
            print(f"  Processing genome {row_idx+1}/{total}...")

        dump_file = os.path.join(DUMP_DIR, f"{gid}.txt")
        with open(dump_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                kmer, count = parts[0], int(parts[1])
                col_idx = vocab.get(kmer)
                if col_idx is not None:
                    rows.append(row_idx)
                    cols.append(col_idx)
                    data.append(count)

    print(f"\n  Building COO matrix...")
    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    data = np.array(data, dtype=np.uint16)

    matrix = coo_matrix((data, (rows, cols)), shape=(n_genomes, n_features))

    print(f"  Converting to CSR...")
    csr = matrix.tocsr()

    print(f"  Shape: {csr.shape}")
    print(f"  Non-zeros: {csr.nnz:,}")
    print(f"  Density: {csr.nnz / (csr.shape[0] * csr.shape[1]) * 100:.4f}%")

    return csr


def main():
    # 1. Get ordered genome list
    print("Loading genome list...")
    genome_ids = get_genome_list()
    print(f"  Genomes with k-mer dumps: {len(genome_ids)}")

    # 2. Pass 1: build vocabulary
    vocab = pass1_build_vocab(genome_ids)

    # 3. Pass 2: build sparse matrix
    csr = pass2_build_matrix(genome_ids, vocab)

    # 4. Save outputs
    print(f"\nSaving outputs...")

    save_npz(MATRIX_PATH, csr)
    print(f"  Matrix: {MATRIX_PATH} ({os.path.getsize(MATRIX_PATH)/1e6:.1f} MB)")

    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Vocab: {VOCAB_PATH} ({os.path.getsize(VOCAB_PATH)/1e6:.1f} MB)")

    with open(ORDER_PATH, "w") as f:
        for gid in genome_ids:
            f.write(gid + "\n")
    print(f"  Genome order: {ORDER_PATH}")

    print(f"\n{'='*60}")
    print(f"  DONE — Matrix shape: {csr.shape}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
