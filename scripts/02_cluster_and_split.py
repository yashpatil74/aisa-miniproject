#!/usr/bin/env python3
"""
Phase 2b: Cluster genomes at 95% ANI and create anti-leakage train/test split.

Input:  data/mash_dist_matrix.tsv  (Mash pairwise distances, tab-delimited)
        data/labels_final.csv      (genome_id, label)
Output: data/split_labels.csv      (genome_id, label, cluster_id, split)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.model_selection import GroupShuffleSplit

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST_MATRIX  = os.path.join(PROJECT_ROOT, "data", "mash_dist_matrix.tsv")
LABELS_CSV   = os.path.join(PROJECT_ROOT, "data", "labels_final.csv")
OUTPUT_CSV   = os.path.join(PROJECT_ROOT, "data", "split_labels.csv")

# ── Parameters ───────────────────────────────────────────────────────────────
ANI_THRESHOLD = 0.05   # Mash distance ≤ 0.05 ≈ 95% ANI
TEST_SIZE     = 0.20   # 80/20 train/test
RANDOM_SEED   = 42


def load_distance_matrix(path):
    """Load Mash tabular distance matrix, return (genome_ids, dist_matrix)."""
    print(f"Loading distance matrix from {path}...")
    df = pd.read_csv(path, sep="\t", index_col=0)

    # Extract genome IDs from filenames (e.g., "fna_files/573.14181.fna" → "573.14181")
    genome_ids = []
    for col in df.columns:
        gid = os.path.basename(col).replace(".fna", "")
        genome_ids.append(gid)

    dist = df.values.astype(np.float32)

    # Ensure symmetric
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)

    print(f"  Matrix shape: {dist.shape}")
    print(f"  Distance range: [{dist[dist > 0].min():.4f}, {dist.max():.4f}]")
    return genome_ids, dist


def cluster_genomes(dist_matrix, threshold):
    """Hierarchical clustering with average linkage."""
    print(f"\nClustering at distance threshold {threshold} (≈{(1-threshold)*100:.0f}% ANI)...")

    # Convert square distance matrix to condensed form
    condensed = squareform(dist_matrix, checks=False)

    # Average linkage clustering
    Z = linkage(condensed, method="average")

    # Cut at threshold
    cluster_ids = fcluster(Z, t=threshold, criterion="distance")

    n_clusters = len(set(cluster_ids))
    print(f"  Clusters formed: {n_clusters}")
    print(f"  Largest cluster: {max(np.bincount(cluster_ids)[1:])} genomes")
    print(f"  Singletons: {sum(1 for c in np.bincount(cluster_ids)[1:] if c == 1)}")

    return cluster_ids


def create_group_split(labels_df, cluster_ids_series, test_size, seed):
    """GroupShuffleSplit ensuring no cluster spans train and test."""
    print(f"\nCreating group-aware split (test_size={test_size})...")

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = cluster_ids_series.values

    train_idx, test_idx = next(gss.split(labels_df, labels_df["label"], groups))

    labels_df = labels_df.copy()
    labels_df["split"] = "train"
    labels_df.loc[labels_df.index[test_idx], "split"] = "test"

    # Verify no cluster leakage
    train_clusters = set(labels_df.loc[labels_df["split"] == "train", "cluster_id"])
    test_clusters  = set(labels_df.loc[labels_df["split"] == "test", "cluster_id"])
    overlap = train_clusters & test_clusters
    assert len(overlap) == 0, f"CLUSTER LEAKAGE: {len(overlap)} clusters in both splits!"

    return labels_df


def main():
    # 1. Load distance matrix
    matrix_ids, dist_matrix = load_distance_matrix(DIST_MATRIX)

    # 2. Load labels
    labels = pd.read_csv(LABELS_CSV)
    labels["genome_id"] = labels["genome_id"].astype(str)
    print(f"\nLabels loaded: {len(labels)} genomes")

    # 3. Align labels with distance matrix ordering
    matrix_id_set = set(matrix_ids)
    label_id_set  = set(labels["genome_id"])

    common = matrix_id_set & label_id_set
    print(f"  Overlap between matrix and labels: {len(common)}")

    if len(common) < len(labels):
        missing = label_id_set - matrix_id_set
        print(f"  WARNING: {len(missing)} labeled genomes not in distance matrix — dropping")

    # Reorder labels to match matrix order, keeping only common
    labels_aligned = labels[labels["genome_id"].isin(common)].copy()
    id_to_idx = {gid: i for i, gid in enumerate(matrix_ids)}
    labels_aligned["_matrix_idx"] = labels_aligned["genome_id"].map(id_to_idx)
    labels_aligned = labels_aligned.sort_values("_matrix_idx").reset_index(drop=True)

    # Subset distance matrix to matching genomes
    keep_idx = labels_aligned["_matrix_idx"].values
    dist_sub = dist_matrix[np.ix_(keep_idx, keep_idx)]

    # 4. Cluster
    cluster_ids = cluster_genomes(dist_sub, ANI_THRESHOLD)
    labels_aligned["cluster_id"] = cluster_ids

    # 5. Group-aware split
    labels_aligned = create_group_split(labels_aligned, labels_aligned["cluster_id"], TEST_SIZE, RANDOM_SEED)

    # 6. Save
    output = labels_aligned[["genome_id", "label", "cluster_id", "split"]]
    output.to_csv(OUTPUT_CSV, index=False)

    # 7. Summary
    train = output[output["split"] == "train"]
    test  = output[output["split"] == "test"]

    print(f"\n{'='*60}")
    print(f"  Saved: {OUTPUT_CSV}")
    print(f"  Total genomes:  {len(output)}")
    print(f"  Total clusters: {output['cluster_id'].nunique()}")
    print(f"")
    print(f"  TRAIN: {len(train)} genomes ({len(train)/len(output)*100:.1f}%)")
    print(f"    Resistant:   {(train['label']==1).sum()}")
    print(f"    Susceptible: {(train['label']==0).sum()}")
    print(f"")
    print(f"  TEST:  {len(test)} genomes ({len(test)/len(output)*100:.1f}%)")
    print(f"    Resistant:   {(test['label']==1).sum()}")
    print(f"    Susceptible: {(test['label']==0).sum()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
