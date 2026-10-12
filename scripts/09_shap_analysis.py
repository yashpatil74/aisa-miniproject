#!/usr/bin/env python3
"""
Phase 5c: SHAP analysis for biological interpretability.

Extracts top 20 most important 31-mer features and prints them
in a BLAST-ready format for searching against CARD.

Input:  models/lgb_sklearn.pkl
        data/kmer_matrix.npz, data/genome_order.txt, data/split_labels.csv
        data/kmer_vocab.pkl
Output: results/shap_top20_kmers.txt
        results/shap_summary.png
        results/top20_kmers.fasta    (BLAST-ready FASTA)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.common

import pickle
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX_PATH  = os.path.join(PROJECT_ROOT, "data", "kmer_matrix.npz")
ORDER_PATH   = os.path.join(PROJECT_ROOT, "data", "genome_order.txt")
SPLIT_CSV    = os.path.join(PROJECT_ROOT, "data", "split_labels.csv")
VOCAB_PATH   = os.path.join(PROJECT_ROOT, "data", "kmer_vocab.pkl")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "lgb_sklearn.pkl")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")

# ── Parameters ───────────────────────────────────────────────────────────────
TOP_N        = 20
MAX_SAMPLES  = 200   # Subsample test set for SHAP (memory management)


def load_test_data():
    """Load test split data."""
    X = load_npz(MATRIX_PATH).astype(np.float32)
    with open(ORDER_PATH, "r") as f:
        genome_order = [line.strip() for line in f]

    split_df = pd.read_csv(SPLIT_CSV)
    split_df["genome_id"] = split_df["genome_id"].astype(str)

    id_to_row = {gid: i for i, gid in enumerate(genome_order)}
    split_df["row_idx"] = split_df["genome_id"].map(id_to_row)
    split_df = split_df.dropna(subset=["row_idx"])
    split_df["row_idx"] = split_df["row_idx"].astype(int)

    test_mask = split_df["split"] == "test"
    test_idx  = split_df.loc[test_mask, "row_idx"].values
    X_test    = X[test_idx]
    y_test    = split_df.loc[test_mask, "label"].values

    return X_test, y_test


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load model
    print("Loading LightGBM model...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # 2. Load test data
    print("Loading test data...")
    X_test, y_test = load_test_data()
    print(f"  Test set: {X_test.shape}")

    # 3. Subsample if needed
    if X_test.shape[0] > MAX_SAMPLES:
        np.random.seed(42)
        idx = np.random.choice(X_test.shape[0], MAX_SAMPLES, replace=False)
        X_shap = X_test[idx]
        print(f"  Subsampled to {MAX_SAMPLES} samples for SHAP")
    else:
        X_shap = X_test

    # 4. Load vocabulary (reverse mapping: column_index → kmer_string)
    print("Loading vocabulary...")
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    idx_to_kmer = {v: k for k, v in vocab.items()}
    print(f"  Vocabulary size: {len(vocab):,}")

    # 5. SHAP analysis
    print("\nRunning SHAP TreeExplainer (this may take a few minutes)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # For binary classification, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # SHAP values for the positive class (Resistant)
    else:
        shap_vals = shap_values

    # 6. Compute mean absolute SHAP per feature
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)

    # Handle sparse output
    if hasattr(mean_abs_shap, "A1"):
        mean_abs_shap = mean_abs_shap.A1

    top_indices = np.argsort(mean_abs_shap)[::-1][:TOP_N]

    # 7. Build results table
    print(f"\n{'='*80}")
    print(f"  TOP {TOP_N} MOST IMPORTANT 31-MER FEATURES (by mean |SHAP|)")
    print(f"{'='*80}")
    header = f"{'Rank':>4}  {'SHAP Importance':>15}  {'31-mer Sequence':<35}  {'Direction'}"
    print(header)
    print("-" * len(header))

    results = []
    for rank, col_idx in enumerate(top_indices, 1):
        importance = mean_abs_shap[col_idx]
        kmer = idx_to_kmer.get(col_idx, f"UNKNOWN_COL_{col_idx}")

        # Determine direction: positive mean SHAP → promotes resistance
        mean_shap = shap_vals[:, col_idx].mean() if not hasattr(shap_vals[:, col_idx], "A1") \
            else shap_vals[:, col_idx].A1.mean()
        direction = "Promotes Resistance" if mean_shap > 0 else "Promotes Susceptibility"

        print(f"{rank:>4}  {importance:>15.6f}  {kmer:<35}  {direction}")
        results.append((rank, importance, kmer, direction))

    print(f"{'='*80}")

    # 8. Save text report
    report_path = os.path.join(RESULTS_DIR, "shap_top20_kmers.txt")
    with open(report_path, "w") as f:
        f.write(f"Top {TOP_N} Most Important 31-mer Features (SHAP TreeExplainer)\n")
        f.write(f"Model: LightGBM\n")
        f.write(f"SHAP samples: {X_shap.shape[0]}\n\n")
        f.write(f"{'Rank':>4}\t{'SHAP Importance':>15}\t{'31-mer Sequence':<35}\t{'Direction'}\n")
        f.write("-" * 90 + "\n")
        for rank, importance, kmer, direction in results:
            f.write(f"{rank:>4}\t{importance:>15.6f}\t{kmer:<35}\t{direction}\n")
    print(f"\nSaved: {report_path}")

    # 9. Save BLAST-ready FASTA
    fasta_path = os.path.join(RESULTS_DIR, "top20_kmers.fasta")
    with open(fasta_path, "w") as f:
        for rank, importance, kmer, direction in results:
            f.write(f">kmer_rank{rank}_shap{importance:.6f}_{direction.replace(' ', '_')}\n")
            f.write(f"{kmer}\n")
    print(f"Saved BLAST-ready FASTA: {fasta_path}")

    # 10. SHAP summary plot (top 20 features)
    print("\nGenerating SHAP summary plot...")
    feature_names = [idx_to_kmer.get(i, f"col_{i}") for i in range(shap_vals.shape[1])]

    # Dense conversion for plotting (only top features)
    if hasattr(X_shap, "toarray"):
        X_plot = X_shap[:, top_indices].toarray()
    else:
        X_plot = X_shap[:, top_indices]

    shap_plot_vals = shap_vals[:, top_indices]
    if hasattr(shap_plot_vals, "toarray"):
        shap_plot_vals = shap_plot_vals.toarray()

    top_names = [idx_to_kmer.get(i, f"col_{i}") for i in top_indices]

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_plot_vals,
        X_plot,
        feature_names=top_names,
        show=False,
        max_display=TOP_N,
    )
    plt.title(f"SHAP Feature Importance — Top {TOP_N} 31-mers", fontsize=14)
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, "shap_summary.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")

    print(f"\n{'='*60}")
    print(f"  DONE — Ready for BLAST against CARD")
    print(f"  Use: results/top20_kmers.fasta")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
