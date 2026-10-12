#!/usr/bin/env python3
"""
Phase 0: Prepare clean, deduplicated labels for the AMR prediction pipeline.

Input:  misc/clean_labels_combined.csv  (synced metadata with possible duplicate rows)
Output: data/labels_final.csv           (genome_id, label — one row per genome)

Label mapping: Susceptible=0, Resistant=1
"""

import os
import sys
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_INPUT    = os.path.join(PROJECT_ROOT, "misc", "clean_labels_combined.csv")
FNA_DIR      = os.path.join(PROJECT_ROOT, "fna_files")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
CSV_OUTPUT   = os.path.join(DATA_DIR, "labels_final.csv")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Load synced metadata
    print(f"Loading metadata from {CSV_INPUT}...")
    df = pd.read_csv(CSV_INPUT)
    df["Genome ID"] = df["Genome ID"].astype(str)
    print(f"  Loaded {len(df)} rows, {df['Genome ID'].nunique()} unique genomes")

    # 2. Map phenotype to binary label
    label_map = {"Susceptible": 0, "Resistant": 1}
    df["label"] = df["Resistant Phenotype"].map(label_map)

    unmapped = df["label"].isna().sum()
    if unmapped > 0:
        print(f"  WARNING: {unmapped} rows have unmapped phenotypes — dropping them")
        df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # 3. Deduplicate on Genome ID (safe: 0 conflicting labels verified)
    df_dedup = df.drop_duplicates(subset="Genome ID", keep="first")
    print(f"  After dedup: {len(df_dedup)} unique genomes")

    # 4. Cross-check against .fna directory
    fna_ids = {f[:-4] for f in os.listdir(FNA_DIR) if f.endswith(".fna")}
    matched = df_dedup[df_dedup["Genome ID"].isin(fna_ids)]
    orphan_csv = set(df_dedup["Genome ID"]) - fna_ids
    orphan_fna = fna_ids - set(df_dedup["Genome ID"])

    if orphan_csv:
        print(f"  WARNING: {len(orphan_csv)} genome IDs in CSV but no .fna file — dropping")
    if orphan_fna:
        print(f"  WARNING: {len(orphan_fna)} .fna files with no CSV label — ignored")

    # 5. Save minimal output
    out = matched[["Genome ID", "label"]].rename(columns={"Genome ID": "genome_id"})
    out = out.sort_values("genome_id").reset_index(drop=True)
    out.to_csv(CSV_OUTPUT, index=False)

    # 6. Summary
    print(f"\n{'='*50}")
    print(f"  Saved: {CSV_OUTPUT}")
    print(f"  Total genomes:  {len(out)}")
    print(f"  Resistant (1):  {(out['label'] == 1).sum()}")
    print(f"  Susceptible (0): {(out['label'] == 0).sum()}")
    print(f"  Class ratio (S/R): {(out['label']==0).sum() / max((out['label']==1).sum(), 1):.2f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
