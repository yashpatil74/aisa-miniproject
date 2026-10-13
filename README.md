# AI-Driven Antimicrobial Resistance (AMR) Prediction from Bacterial Whole Genomes

**Target Organism:** *Klebsiella pneumoniae*  
**Target Antibiotic:** Meropenem (Carbapenem class)  
**Task:** Binary Classification (Susceptible vs. Resistant)  
**Methodology:** Alignment-free 31-mer tokenization, MinHash clonal clustering, High-dimensional Sparse CSR matrices, Gradient Boosted Decision Trees (LightGBM & XGBoost), and SHAP biological interpretability.

---

## 1. Executive Summary & Clinical Context

### The Core Problem: The Diagnostic Bottleneck
Antimicrobial Resistance (AMR) is recognized by the World Health Organization (WHO) as one of the top ten global public health threats. *Klebsiella pneumoniae* is a high-priority Gram-negative pathogen responsible for severe hospital-acquired pneumonia, bloodstream infections, and sepsis.

**Meropenem** is an ultra-broad-spectrum carbapenem antibiotic considered a clinical "drug of last resort." Resistance to meropenem leaves clinicians with few, highly toxic alternatives (e.g., colistin).

Traditional diagnostic identification relies on culture-based **Antimicrobial Susceptibility Testing (AST)** (broth microdilution or disk diffusion). This process requires **24 to 72 hours** to culture and test the bacteria:
* In cases of septic shock, patient mortality increases by ~7.6% for every hour of delayed effective antibiotic treatment.
* Clinicians are often forced to prescribe empiric broad-spectrum antibiotics, exacerbating resistance selection.

### The AI Solution
With next-generation Whole Genome Sequencing (WGS) becoming rapid and cost-effective, bacterial DNA can be sequenced directly from clinical samples in hours. This pipeline trains machine learning models to analyze the raw, continuous DNA sequence of an isolate and **predict phenotypic resistance directly from the genome on day one**, bypassing the culture bottleneck.

---

## 2. Theoretical Foundations: What and Why

### Why Alignment-Free K-mer Tokenization?
Traditional bioinformatics aligns reads against a single reference genome (e.g., *K. pneumoniae* MGH 78578). However:
1. **Mobile Genetic Elements (MGEs):** Major carbapenemase resistance genes (*blaKPC*, *blaNDM*, *blaOXA-48*) reside on plasmids and transposons that undergo extensive horizontal gene transfer, insertion, and structural rearrangement. Reference-based alignment often discards or misaligns unmapped plasmid sequences.
2. **Alignment-Free K-mers:** By breaking the 5.5-million-base-pair genome into overlapping substrings of length $k$ (k-mers), the model detects the presence or absence of resistance-conferring signatures regardless of genomic position or structural rearrangements.

### Why 31-mers ($k=31$)?
1. **Sequence Specificity:** For a 5.5 Mbp bacterial genome, $k=31$ is mathematically long enough to guarantee that almost all k-mers are unique within the genome, preventing false positive matches between unrelated genes.
2. **Hardware Register Efficiency:** A DNA base (`A`, `C`, `G`, `T`) requires 2 bits ($2^2 = 4$). A 31-mer sequence requires:
   $$31 \times 2 = 62\text{ bits}$$
   This fits inside a standard **64-bit CPU register**, enabling fast bitwise operations during counting in tools like KMC.

### Why Clonal Clustering & Anti-Leakage Splitting?
Bacteria reproduce clonally via binary fission. A single hospital outbreak can result in dozens of virtually identical bacterial isolates being deposited into surveillance databases.

* **The Clonal Leakage Trap:** A naive random 80/20 train/test split would place clones of the same outbreak strain into both the training and testing sets. The machine learning model would memorize background genomic mutations specific to that hospital strain rather than true resistance mechanisms.
* **The Solution:** We use **Mash (MinHash sketching)** to compute pairwise genetic distances across all genomes, cluster them at $\ge 99.9\%$ Average Nucleotide Identity (ANI) using hierarchical clustering, and perform a **Grouped Split (`GroupShuffleSplit`)**. All members of a clonal outbreak are placed strictly into Train or strictly into Test.

### Why SciPy Sparse CSR Matrices?
* The dataset contains **3,972 genomes** and over **2.1 million unique 31-mers**.
* A dense 32-bit float array of this size would require:
  $$3,972 \times 2,132,407 \times 4\text{ bytes} \approx 33.9\text{ GB of RAM}$$
* However, any individual bacterial genome only contains a small fraction of the global vocabulary ($>98\%$ sparse).
* By storing features in **Compressed Sparse Row (CSR)** format (`scipy.sparse.csr_matrix`), RAM footprint is reduced to **~100 MB**, allowing training on standard hardware.

### Why Gradient Boosted Decision Trees (LightGBM & XGBoost)?
* **Curse of Dimensionality:** Deep neural networks rapidly overfit when feature count ($2.1\times 10^6$) vastly exceeds sample size ($3.9\times 10^3$).
* **Sparsity-Aware Tree Splitting:** Tree algorithms branch on feature splits without requiring dense linear projections, handling extreme sparsity naturally.
* **Class Imbalance Support:** Clinical resistance datasets are inherently imbalanced. Gradient boosting natively accepts `scale_pos_weight` to penalize False Negatives (missing a resistant strain).

---

## 3. End-to-End Pipeline Architecture

```
                       Raw Data (misc/clean_labels_combined.csv, fna_files/*.fna)
                                                  │
                                                  ▼
                                      [02_prepare_labels.py]
                                (Deduplicate & binary label mapping)
                                                  │
                                                  ▼
                                       [03_mash_sketch.sh]
                                  (MinHash pairwise distances)
                                                  │
                                                  ▼
                                      [04_group_and_split.py]
                                (Hierarchical clustering at 99.9% ANI
                                     + GroupShuffleSplit 80/20)
                                                  │
                                                  ▼
                                       [05_count_kmers.sh]
                                    (KMC 31-mer extraction)
                                                  │
                                                  ▼
                                   [06_build_feature_matrix.py]
                                  (Two-pass SciPy CSR matrix)
                                                  │
                                                  ▼
                                      [07_train_model.py]
                             (Train LightGBM & XGBoost with class weights)
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         ▼                                                 ▼
                [08_evaluate.py]                                  [09_shap_analysis.py]
        (Macro F1, MCC, ROC/PR-AUC,                         (TreeExplainer top 20 k-mers,
         threshold optimization)                                 CARD FASTA export)
                         │
                         ▼
               [10_predict_genome.py]
           (Live CLI inference on new .fna)
```

---

## 4. Repository Structure

```
├── data/
│   ├── genome_order.txt          # Ordering of genomes in sparse matrix rows
│   ├── kmer_matrix.npz           # SciPy CSR sparse feature matrix (3,972 x 2.1M)
│   ├── kmer_vocab.pkl            # Pickled mapping: 31-mer string -> column index
│   ├── labels_final.csv          # Clean binary labels (Susceptible=0, Resistant=1)
│   ├── mash_dist_matrix.tsv      # 3,972 x 3,972 pairwise Mash genetic distances
│   └── split_labels.csv          # Train/Test assignment with cluster IDs
├── fna_files/                    # Multi-FASTA (.fna) Whole Genome Shotgun contigs
├── misc/                         # Raw BV-BRC metadata downloads and sync scripts
├── models/
│   ├── lightgbm_model.txt        # Native LightGBM booster model
│   ├── lgb_sklearn.pkl           # Scikit-learn wrapper for LightGBM
│   ├── xgboost_model.json        # Native XGBoost model
│   └── xgb_sklearn.pkl           # Scikit-learn wrapper for XGBoost
├── results/
│   ├── confusion_matrix_lightgbm.png        # Confusion matrix plot for LightGBM
│   ├── confusion_matrix_xgboost.png         # Confusion matrix plot for XGBoost
│   ├── evaluation_report.txt                # Detailed evaluation metrics (AUC, F1, MCC)
│   ├── plot1_roc_and_pr_curves.png          # Clinical ROC and Precision-Recall curves
│   ├── plot2_clonal_distance_ani_distribution.png # 95% vs 99.9% ANI distance distribution
│   ├── plot3_model_benchmark_comparison.png # Before vs After model performance benchmark
│   ├── plot4_clinical_threshold_tuning.png  # Sensitivity/Specificity threshold trade-offs
│   ├── plot5_top20_shap_kmers.png           # Ranked top 20 SHAP features with direction
│   ├── shap_summary.png                     # SHAP beeswarm importance plot
│   ├── shap_top20_kmers.txt                 # Ranked table of top 20 31-mers
│   └── top20_kmers.fasta                    # BLAST-ready FASTA of top predictive k-mers
└── scripts/
    ├── common.py                 # Shared environment & OpenMP runtime loader
    ├── 02_prepare_labels.py      # Phase 1: Label deduplication and cleaning
    ├── 03_mash_sketch.sh         # Phase 2: Mash sketching and distance matrix
    ├── 04_group_and_split.py     # Phase 3: Clonal clustering & GroupShuffleSplit
    ├── 05_count_kmers.sh         # Phase 4: Parallel KMC 31-mer extraction
    ├── 06_build_feature_matrix.py# Phase 5: Two-pass SciPy sparse matrix assembly
    ├── 07_train_model.py         # Phase 6: Model training (LightGBM & XGBoost)
    ├── 08_evaluate.py            # Phase 7: Model evaluation & threshold analysis
    ├── 09_shap_analysis.py       # Phase 8: SHAP analysis & CARD export
    ├── 10_predict_genome.py      # Phase 9: CLI inference for unseen genomes
    └── 11_generate_plots.py      # Phase 10: Publication-grade figures (300 DPI)
```

---

## 5. How to Run the Pipeline

### Step 0: Activate Environment
```bash
cd /home/yash/college/pe1/miniproject
source .venv/bin/activate
```

### Step 1: Label Preparation
```bash
python scripts/02_prepare_labels.py
```
*Outputs `data/labels_final.csv` with 3,972 genomes (1,502 Resistant, 2,470 Susceptible).*

### Step 2: Mash Distance Matrix
```bash
bash scripts/03_mash_sketch.sh
```
*Outputs `data/mash_dist_matrix.tsv`. (Can be skipped if the matrix is already present in `data/`).*

### Step 3: Anti-Leakage Clonal Splitting
```bash
python scripts/04_group_and_split.py
```
*Clusters strains at 99.9% ANI and splits into Train (2,993 isolates) and Test (979 isolates).*

### Step 4: K-mer Extraction & Matrix Assembly
```bash
bash scripts/05_count_kmers.sh
python scripts/06_build_feature_matrix.py
```
*Builds the 2,132,407-feature sparse CSR matrix. (Can be skipped if `data/kmer_matrix.npz` already exists).*

### Step 5: Model Training
```bash
python scripts/07_train_model.py
```
*Trains LightGBM and XGBoost classifiers with `scale_pos_weight`.*

### Step 6: Evaluation
```bash
python scripts/08_evaluate.py
```
*Generates `results/evaluation_report.txt` and confusion matrix plots.*

### Step 7: Biological Interpretability (SHAP)
```bash
python scripts/09_shap_analysis.py
```
*Generates SHAP summary plots and `results/top20_kmers.fasta` ready for BLAST searching.*

### Step 8: Live Inference on an Unseen Genome
```bash
python scripts/10_predict_genome.py path/to/sample.fna
```
*Prints resistance status and calibrated confidence score.*

### Step 9: Generate Publication-Grade Visualizations
```bash
python scripts/11_generate_plots.py
```
*Generates all 5 high-resolution figures (`plot1` to `plot5`) in `results/`.*

---

## 6. Evaluation Metrics Explained

In clinical AMR diagnostic models, **Raw Accuracy is strictly misleading**:
* A naive model predicting all samples as "Susceptible" on an imbalanced dataset would achieve high accuracy while failing to detect life-threatening resistant infections.
* The FDA and Clinical & Laboratory Standards Institute (CLSI) classify False Susceptibles as **Very Major Errors (VME)** (fatal risk of untreated infection) and False Resistants as **Major Errors (ME)** (overuse of reserve antibiotics).

### Reported Metrics:
1. **Macro F1-Score:** Harmonic mean of precision and recall computed independently for both classes and averaged, penalizing poor performance on the minority class.
2. **Matthews Correlation Coefficient (MCC):** Range $[-1, +1]$. Widely considered the gold standard metric for imbalanced binary classification.
3. **ROC-AUC & PR-AUC:** Area under Receiver Operating Characteristic and Precision-Recall curves, evaluating classifier discrimination across all potential decision thresholds.
4. **Optimal Threshold Tuning:** Identifies the operating cutoff that balances clinical sensitivity and specificity.

---

## 7. Biological Validation (The "Black-Box" Fix)

Machine learning models in healthcare must provide biological explainability:
1. **SHAP TreeExplainer:** Computes exact Shapley values from cooperative game theory to quantify the marginal contribution of each 31-mer to the positive (Resistant) prediction.
2. **BLAST against CARD:** The top 20 predictive 31-mers in `results/top20_kmers.fasta` can be aligned against the **Comprehensive Antibiotic Resistance Database (CARD)** or NCBI BLAST.
3. **Biological Alignment:** True positive features align with known carbapenemase genes (*blaKPC-2*, *blaKPC-3*, *blaNDM-1*, *blaOXA-48*) and outer membrane porin disruptions (*ompK35*, *ompK36*), proving the model reverse-engineered real-world molecular mechanisms without human biological supervision.
