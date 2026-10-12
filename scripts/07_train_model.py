#!/usr/bin/env python3
"""
Phase 5a: Train LightGBM and XGBoost models for AMR prediction.

Input:  data/kmer_matrix.npz     (scipy CSR sparse matrix)
        data/genome_order.txt    (row ordering)
        data/split_labels.csv    (genome_id, label, cluster_id, split)
Output: models/lightgbm_model.txt
        models/xgboost_model.json
"""

import os
import sys

# Ensure library path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.common

import pickle
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import lightgbm as lgb
import xgboost as xgb

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX_PATH  = os.path.join(PROJECT_ROOT, "data", "kmer_matrix.npz")
ORDER_PATH   = os.path.join(PROJECT_ROOT, "data", "genome_order.txt")
SPLIT_CSV    = os.path.join(PROJECT_ROOT, "data", "split_labels.csv")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")

RANDOM_SEED = 42


def load_data():
    """Load and align the sparse matrix with train/test split labels."""
    print("Loading data...")

    # Load sparse matrix
    X = load_npz(MATRIX_PATH).astype(np.float32)
    print(f"  Matrix shape: {X.shape}")

    # Load genome ordering
    with open(ORDER_PATH, "r") as f:
        genome_order = [line.strip() for line in f]
    print(f"  Genome order: {len(genome_order)} entries")

    # Load split labels
    split_df = pd.read_csv(SPLIT_CSV)
    split_df["genome_id"] = split_df["genome_id"].astype(str)
    print(f"  Split labels: {len(split_df)} entries")

    # Create genome_id → row_index mapping from matrix order
    id_to_row = {gid: i for i, gid in enumerate(genome_order)}

    # Align: find matrix row index for each genome in split_labels
    split_df["row_idx"] = split_df["genome_id"].map(id_to_row)
    split_df = split_df.dropna(subset=["row_idx"])
    split_df["row_idx"] = split_df["row_idx"].astype(int)

    # Split
    train_mask = split_df["split"] == "train"
    test_mask  = split_df["split"] == "test"

    train_idx = split_df.loc[train_mask, "row_idx"].values
    test_idx  = split_df.loc[test_mask,  "row_idx"].values

    X_train = X[train_idx]
    X_test  = X[test_idx]
    y_train = split_df.loc[train_mask, "label"].values
    y_test  = split_df.loc[test_mask,  "label"].values

    print(f"\n  Train: {X_train.shape[0]} samples  (R={y_train.sum()}, S={(y_train==0).sum()})")
    print(f"  Test:  {X_test.shape[0]} samples  (R={y_test.sum()}, S={(y_test==0).sum()})")

    return X_train, X_test, y_train, y_test


def train_lightgbm(X_train, y_train, scale_pos_weight):
    """Train LightGBM classifier."""
    print(f"\n{'='*60}")
    print("Training LightGBM...")
    print(f"{'='*60}")

    model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        scale_pos_weight=scale_pos_weight,
        num_leaves=63,
        learning_rate=0.05,
        n_estimators=500,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.3,
        random_state=RANDOM_SEED,
        verbose=-1,
        n_jobs=4,
        max_bin=63,
        device="cpu",
    )

    model.fit(X_train, y_train)
    print("  Training complete.")
    return model


def train_xgboost(X_train, y_train, scale_pos_weight):
    """Train XGBoost classifier."""
    print(f"\n{'='*60}")
    print("Training XGBoost...")
    print(f"{'='*60}")

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.3,
        random_state=RANDOM_SEED,
        n_jobs=4,
        max_bin=63,
        verbosity=0,
        tree_method="hist",
    )

    model.fit(X_train, y_train)
    print("  Training complete.")
    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load data
    X_train, X_test, y_train, y_test = load_data()

    # 2. Compute class weight
    n_susceptible = (y_train == 0).sum()
    n_resistant   = (y_train == 1).sum()
    spw = n_susceptible / max(n_resistant, 1)
    print(f"\n  scale_pos_weight: {spw:.3f}")

    # 3. Train models
    lgb_model = train_lightgbm(X_train, y_train, spw)
    xgb_model = train_xgboost(X_train, y_train, spw)

    # 4. Save models
    lgb_path = os.path.join(MODEL_DIR, "lightgbm_model.txt")
    lgb_model.booster_.save_model(lgb_path)
    print(f"\n  Saved LightGBM: {lgb_path}")

    xgb_path = os.path.join(MODEL_DIR, "xgboost_model.json")
    xgb_model.save_model(xgb_path)
    print(f"  Saved XGBoost:  {xgb_path}")

    # Also save the sklearn wrapper for easier loading in evaluation
    with open(os.path.join(MODEL_DIR, "lgb_sklearn.pkl"), "wb") as f:
        pickle.dump(lgb_model, f)
    with open(os.path.join(MODEL_DIR, "xgb_sklearn.pkl"), "wb") as f:
        pickle.dump(xgb_model, f)

    # 5. Quick preview
    lgb_pred = lgb_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)

    from sklearn.metrics import f1_score, matthews_corrcoef
    print(f"\n{'='*60}")
    print(f"  Quick Preview (test set):")
    print(f"  LightGBM — Macro F1: {f1_score(y_test, lgb_pred, average='macro'):.4f}  "
          f"MCC: {matthews_corrcoef(y_test, lgb_pred):.4f}")
    print(f"  XGBoost  — Macro F1: {f1_score(y_test, xgb_pred, average='macro'):.4f}  "
          f"MCC: {matthews_corrcoef(y_test, xgb_pred):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
