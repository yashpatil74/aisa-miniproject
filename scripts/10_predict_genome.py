#!/usr/bin/env python3
"""
Phase 6: Live Demo / Inference Script

Takes a new, unseen K. pneumoniae genome (.fna), extracts its 31-mers,
and predicts Meropenem resistance using the trained LightGBM model.

Usage:
    python3 scripts/08_predict_genome.py path/to/genome.fna
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.common

import subprocess
import pickle
import tempfile
import numpy as np
from scipy.sparse import coo_matrix

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB_PATH   = os.path.join(PROJECT_ROOT, "data", "kmer_vocab.pkl")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "xgb_sklearn.pkl")

def extract_kmers(fna_path, output_txt):
    """Run KMC to extract 31-mers from the input genome."""
    print(f"[*] Extracting 31-mers using KMC from: {fna_path}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        kmc_db = os.path.join(tmpdir, "kmer_db")
        
        # Run KMC
        cmd_kmc = [
            "kmc", "-k31", "-fm", "-ci2", "-cs65535", "-t4",
            fna_path, kmc_db, tmpdir
        ]
        subprocess.run(cmd_kmc, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # Dump to text
        cmd_dump = [
            "kmc_tools", "transform", kmc_db, "dump", output_txt
        ]
        subprocess.run(cmd_dump, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def vectorize_genome(dump_txt, vocab):
    """Convert the KMC text dump into a 1 x N sparse matrix using the vocabulary."""
    print("[*] Vectorizing genome against trained vocabulary...")
    n_features = len(vocab)
    
    cols = []
    data = []
    
    with open(dump_txt, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                kmer, count = parts[0], int(parts[1])
                if kmer in vocab:
                    cols.append(vocab[kmer])
                    data.append(count)
    
    rows = np.zeros(len(cols), dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    data = np.array(data, dtype=np.float32)
    
    csr = coo_matrix((data, (rows, cols)), shape=(1, n_features)).tocsr()
    print(f"    - Genome contains {len(cols):,} / {n_features:,} known k-mers")
    return csr

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/08_predict_genome.py path/to/genome.fna")
        sys.exit(1)
        
    fna_path = sys.argv[1]
    if not os.path.isfile(fna_path):
        print(f"Error: File not found -> {fna_path}")
        sys.exit(1)
        
    # 1. Load Vocab and Model
    print("[*] Loading vocabulary and model...")
    if not os.path.isfile(VOCAB_PATH) or not os.path.isfile(MODEL_PATH):
        print("Error: Vocabulary or Model not found. Run training pipeline first.")
        sys.exit(1)
        
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # 2. Extract and Vectorize
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_dump:
        dump_txt = tmp_dump.name
        
    try:
        extract_kmers(fna_path, dump_txt)
        X_new = vectorize_genome(dump_txt, vocab)
        
        # 3. Predict
        print("[*] Running inference...")
        prob = model.predict_proba(X_new)[0][1]
        prediction = "Resistant" if prob >= 0.5 else "Susceptible"
        
        print("\n" + "="*50)
        print(f"  PREDICTION RESULTS")
        print("="*50)
        print(f"  Genome:     {os.path.basename(fna_path)}")
        print(f"  Status:     {prediction}")
        print(f"  Confidence: {prob*100:.1f}% probability of Resistance")
        print("="*50 + "\n")
        
    finally:
        if os.path.exists(dump_txt):
            os.remove(dump_txt)

if __name__ == "__main__":
    main()
