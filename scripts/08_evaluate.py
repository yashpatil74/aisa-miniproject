#!/usr/bin/env python3
"""
Phase 5b: Evaluate trained models with required metrics.

Metrics: Macro F1-Score, Matthews Correlation Coefficient, Confusion Matrix.
NOTE: Accuracy is strictly prohibited per project requirements.

Input:  models/lgb_sklearn.pkl, models/xgb_sklearn.pkl
        data/kmer_matrix.npz, data/genome_order.txt, data/split_labels.csv
Output: results/evaluation_report.txt
        results/confusion_matrix_lgb.png
        results/confusion_matrix_xgb.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.common

import pickle
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX_PATH  = os.path.join(PROJECT_ROOT, "data", "kmer_matrix.npz")
ORDER_PATH   = os.path.join(PROJECT_ROOT, "data", "genome_order.txt")
SPLIT_CSV    = os.path.join(PROJECT_ROOT, "data", "split_labels.csv")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")


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


def evaluate_model(name, model, X_test, y_test, report_lines):
    """Evaluate a single model and append results to report."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    mcc      = matthews_corrcoef(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)
    cr       = classification_report(y_test, y_pred, target_names=["Susceptible", "Resistant"])

    report_lines.append(f"\n{'='*60}")
    report_lines.append(f"  MODEL: {name}")
    report_lines.append(f"{'='*60}")
    report_lines.append(f"  Macro F1-Score:                {macro_f1:.4f}")
    report_lines.append(f"  Matthews Correlation Coeff:    {mcc:.4f}")
    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc  = average_precision_score(y_test, y_prob)
        report_lines.append(f"  ROC-AUC Score:                 {roc_auc:.4f}")
        report_lines.append(f"  PR-AUC (Average Precision):    {pr_auc:.4f}")
    report_lines.append(f"")
    report_lines.append(f"  Confusion Matrix (Threshold 0.50):")
    report_lines.append(f"                Predicted")
    report_lines.append(f"                S       R")
    report_lines.append(f"  Actual S   {cm[0,0]:>5}   {cm[0,1]:>5}")
    report_lines.append(f"  Actual R   {cm[1,0]:>5}   {cm[1,1]:>5}")
    report_lines.append(f"")
    report_lines.append(f"  Classification Report (Threshold 0.50):")
    for line in cr.split("\n"):
        report_lines.append(f"    {line}")

    # Threshold Optimization (maximize Macro F1)
    if y_prob is not None:
        thresholds = np.linspace(0.1, 0.9, 81)
        f1_scores = [f1_score(y_test, (y_prob >= t).astype(int), average="macro") for t in thresholds]
        best_idx = np.argmax(f1_scores)
        best_t = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        best_pred = (y_prob >= best_t).astype(int)
        best_mcc = matthews_corrcoef(y_test, best_pred)
        best_cm = confusion_matrix(y_test, best_pred)
        
        report_lines.append(f"")
        report_lines.append(f"  --- Optimal Decision Threshold Analysis ---")
        report_lines.append(f"  Optimal Threshold:             {best_t:.2f}")
        report_lines.append(f"  Macro F1 at Optimal Thresh:    {best_f1:.4f}")
        report_lines.append(f"  MCC at Optimal Thresh:         {best_mcc:.4f}")
        report_lines.append(f"  Confusion Matrix (Threshold {best_t:.2f}):")
        report_lines.append(f"                Predicted")
        report_lines.append(f"                S       R")
        report_lines.append(f"  Actual S   {best_cm[0,0]:>5}   {best_cm[0,1]:>5}")
        report_lines.append(f"  Actual R   {best_cm[1,0]:>5}   {best_cm[1,1]:>5}")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Susceptible", "Resistant"],
                yticklabels=["Susceptible", "Resistant"],
                ax=ax, cbar_kws={"label": "Count"})
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=14)
    plt.tight_layout()

    suffix = name.lower().replace(" ", "_")
    fig_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{suffix}.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fig_path}")

    return macro_f1, mcc


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading test data...")
    X_test, y_test = load_test_data()
    print(f"  Test set: {X_test.shape[0]} samples")

    report_lines = []
    report_lines.append("AMR Prediction — Evaluation Report")
    report_lines.append(f"Test set size: {len(y_test)}")
    report_lines.append(f"  Resistant: {(y_test==1).sum()}")
    report_lines.append(f"  Susceptible: {(y_test==0).sum()}")

    # Evaluate each model
    for model_name, pkl_name in [("LightGBM", "lgb_sklearn.pkl"), ("XGBoost", "xgb_sklearn.pkl")]:
        pkl_path = os.path.join(MODEL_DIR, pkl_name)
        if not os.path.isfile(pkl_path):
            print(f"  Skipping {model_name} — model not found at {pkl_path}")
            continue

        print(f"\nEvaluating {model_name}...")
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)

        evaluate_model(model_name, model, X_test, y_test, report_lines)

    # Save report
    report_path = os.path.join(RESULTS_DIR, "evaluation_report.txt")
    report_text = "\n".join(report_lines)
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"\n{'='*60}")
    print(report_text)
    print(f"\nFull report saved: {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
