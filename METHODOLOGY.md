# AMR Prediction: Technical Methodology & Implementation Guide

This document provides a comprehensive, deep-dive examination of the biological, mathematical, and computational techniques implemented in the AI-driven Antimicrobial Resistance (AMR) prediction pipeline.

---

## Table of Contents
1. [Biological Background: Meropenem Resistance in Klebsiella pneumoniae](#1-biological-background)
2. [Data Acquisition & Quality Control (QC) Protocols](#2-data-acquisition--qc)
3. [Preventing Clonal Data Leakage: MinHash & Taxonomic vs Clonal Thresholds](#3-preventing-clonal-data-leakage)
4. [Feature Engineering: Alignment-Free K-mer Extraction](#4-feature-engineering)
5. [Data Architecture: Memory-Safe SciPy Sparse CSR Matrices](#5-data-architecture)
6. [Machine Learning Models: GBDTs vs Deep Learning](#6-machine-learning-models)
7. [Clinical Imbalance & Diagnostic Evaluation Metrics](#7-diagnostic-evaluation-metrics)
8. [Biological Explainability: SHAP Game Theory & CARD Validation](#8-biological-explainability)

---

## 1. Biological Background: Meropenem Resistance in *Klebsiella pneumoniae*

### The Pathogen
*Klebsiella pneumoniae* is an encapsulated Gram-negative bacterium belonging to the order *Enterobacterales*. It is an opportunistic human pathogen frequently causing:
* Hospital-acquired ventilator-associated pneumonia (VAP)
* Catheter-associated urinary tract infections (CAUTI)
* Primary pyogenic liver abscesses
* High-mortality bacteremia and septic shock in immunocompromised patients

### The Antibiotic: Meropenem
Meropenem is a **carbapenem** $\beta$-lactam antibiotic. Carbapenems contain a fused $\beta$-lactam ring and a carbapenem core that renders them exceptionally resistant to hydrolysis by standard bacterial $\beta$-lactamases (such as TEM, SHV, and CTX-M extended-spectrum $\beta$-lactamases / ESBLs). Meropenem works by binding to **Penicillin-Binding Proteins (PBPs)** (specifically PBP-2 and PBP-3), preventing peptidoglycan cell wall cross-linking and triggering rapid bacterial cell lysis.

### Mechanisms of Resistance
Resistance in *K. pneumoniae* to meropenem emerges via three distinct molecular routes:

```
                          Meropenem Resistance Mechanisms
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
1. Carbapenemase Production     2. Porin Disruption             3. Efflux Pump
   - Serine (blaKPC, blaOXA-48)    - ompK35 loss (stop codons)     - acrAB overexpression
   - Metallo (blaNDM, blaVIM)      - ompK36 loop 3 insertions      - rarA, ramA upregulation
```

1. **Acquisition of Carbapenemase Enzymes (Primary Driver):**
   * **Class A Serine Carbapenemases (*blaKPC-2*, *blaKPC-3*):** Hydrolyze carbapenems using an active-site serine. Transferred horizontally via the `Tn4401` transposon on conjugative plasmids (e.g., pKpQIL). Highly prevalent in the ST258 and ST11 pandemic clones.
   * **Class B Metallo-$\beta$-Lactamases (*blaNDM-1*, *blaNDM-5*, *blaVIM*, *blaIMP*):** Utilize divalent zinc ions ($Zn^{2+}$) to activate a water molecule that cleaves the $\beta$-lactam ring. Immune to standard $\beta$-lactamase inhibitors like avibactam.
   * **Class D Oxacillinases (*blaOXA-48*, *blaOXA-181*, *blaOXA-232*):** Weaker carbapenem hydrolysis, but when combined with porin loss, causes clinical meropenem resistance.
2. **Outer Membrane Permeability Reduction (Porin Loss):**
   * Meropenem enters the Gram-negative periplasm through non-specific outer membrane channels: **OmpK35** and **OmpK36**.
   * Mutations, transposon insertions, or premature stop codons in *ompK35* or amino acid duplications in the loop L3 region of *ompK36* (e.g., Gly134-Asp135 duplication) restrict drug influx.
3. **Active Efflux:**
   * Upregulation of Resistance-Nodulation-Division (RND) family efflux systems (**AcrAB-TolC** and **OqxAB**) pumps intracellular meropenem back across the outer membrane.

---

## 2. Data Acquisition & Quality Control (QC) Protocols

### BV-BRC Ground Truth Curation
Data is sourced from the **Bacterial and Viral Bioinformatics Resource Center (BV-BRC)**, linking Whole Genome Shotgun (WGS) multi-FASTA files (`.fna`) with clinical laboratory Antimicrobial Susceptibility Testing (AST) metadata.

1. **Definitive Label Filtering:**
   * Only isolates with confirmed phenotypic resistance classifications (`Susceptible` or `Resistant`) are retained.
   * **Intermediate** strains are dropped. Clinical breakpoints for carbapenems have shifted over time, and intermediate phenotypes often represent heterogeneous micro-colonies. Removing them ensures the model trains against clean biological signals.
   * Binary encoding: $y_i \in \{0, 1\}$, where Susceptible = $0$ and Resistant = $1$.

2. **Genomic Quality Control Checks:**
   * **Assembly Completeness:** Genomes with file sizes $<1\text{ MB}$ are dropped (standard *K. pneumoniae* genome is $\approx 5.3 - 5.8\text{ MB}$).
   * **Header Integrity:** Assemblies must begin with valid FASTA definitions (`>`).
   * **Assembly Fragmentation:** Genomes fragmented into $>500$ contigs are excluded to avoid highly corrupted or low-coverage sequence reads.

---

## 3. Preventing Clonal Data Leakage: MinHash & Genomic Distances

### The Problem: Outbreak Duplication Bias
Bacterial transmission often involves hospital clonal outbreaks. For example, a single hospital intensive care unit may yield 60 isolates from a single clone of *K. pneumoniae* ST258.

If standard random 80/20 train/test splitting is used:
* Identical outbreak clones appear in both the training set and testing set.
* The classifier achieves near-perfect test scores simply by **memorizing background phylogenetic mutations** (e.g., neutral ribosomal or metabolic SNPs unique to that hospital lineage) without learning the causal resistance machinery.
* When deployed on a clinical isolate from an unseen hospital or lineage, model performance collapses.

### MinHash (Mash) Pairwise Distance Calculation
Comparing 4,000 whole genomes using full pairwise BLAST or Smith-Waterman alignment is computationally intractable ($O(N^2 \cdot L^2)$). Instead, the **Mash** algorithm reduces genomes to small, fixed-size MinHash sketches:

1. Extract all canonical 21-mers ($k=21$) from each genome.
2. Hash each k-mer using a 64-bit murmur3 hash function: $h(s) \to [0, 2^{64}-1]$.
3. Keep the smallest $s = 10,000$ hash values: $S(G) = \min_s \{h(k_i)\}$.
4. Estimate the Jaccard similarity $J$ between two genomes $A$ and $B$:
   $$J(A, B) = \frac{|S(A) \cap S(B)|}{s}$$
5. Transform Jaccard similarity into Mash distance $D$, which approximates Average Nucleotide Identity (ANI):
   $$D = -\frac{1}{k} \ln\left(\frac{2J}{1 + J}\right), \quad \text{where } \text{ANI} \approx 1 - D$$

### Critical Insight: 95% ANI vs 99.9% ANI Thresholds
* **The 95% ANI Species Barrier:** In prokaryotic taxonomy, **95% ANI** ($D = 0.05$) is the universal boundary defining whether two bacteria belong to the **same species**. Because all 3,972 genomes in this project belong to *Klebsiella pneumoniae*, setting `ANI_THRESHOLD = 0.05` causes **98.1% of the entire dataset (3,896 genomes) to collapse into a single monolithic cluster**.
* **The 99.9% Clonal Outbreak Threshold:** Clonal outbreak lineages and sequence types (STs) typically share **$>99.8\%$ to $99.9\%$ ANI** ($D \le 0.001 - 0.002$), differing by fewer than a few thousand SNPs.
* Setting `ANI_THRESHOLD = 0.001` separates the 3,972 genomes into **1,580 well-defined clonal groups**.
* When passed to `GroupShuffleSplit(test_size=0.20)`, this produces a representative **2,993-sample training set** and **979-sample testing set** (with 309 resistant and 670 susceptible strains), providing strong statistical power while completely preventing clonal leakage.

---

## 4. Feature Engineering: Alignment-Free K-mer Extraction

### Why K-mers Over Reference Alignment?
In bacterial genomics, antimicrobial resistance genes are frequently carried on **plasmids** and **integrons** that are absent in standard reference genomes (like MGH 78578). Read alignment against a reference discards these critical plasmid sequences as "unmapped."

K-mer counting is completely **alignment-free**:
* The input DNA text is scanned with a sliding window of length $k=31$.
* Canonical representation: for each 31-mer, its reverse complement is computed, and the lexicographically smaller sequence is recorded. This ensures forward and reverse strand reads map to the identical token.

### Mathematical Tokenization with KMC
The **KMC (K-mer Counter)** tool is used for multi-threaded disk-based k-mer frequency counting:
```bash
kmc -k31 -fm -ci2 -cs65535 -t1 input.fna tmp_db/ tmp/
```
* `-k31`: Length of 31 base pairs (fits inside a 64-bit integer: $31 \times 2 = 62\text{ bits}$).
* `-ci2`: Minimum occurrence count of 2 per genome (filters out single-read sequencing artifacts).
* `-cs65535`: Counts capped at $2^{16}-1$, stored as compact `uint16` values.

---

## 5. Data Architecture: Memory-Safe SciPy Sparse CSR Matrices

### The High-Dimensional Memory Challenge
* Number of samples ($N$): **3,972**
* Number of unique 31-mers ($M$): **2,132,407**
* If represented as a standard dense float32 array or Pandas DataFrame:
  $$\text{RAM} = 3,972 \times 2,132,407 \times 4\text{ bytes} \approx 33.9\text{ Gigabytes}$$
  This would instantly crash standard computing environments.

### The Two-Pass Sparse Assembly Strategy
To safely construct the matrix within 16 GB RAM limits, [scripts/06_build_feature_matrix.py](file:///home/yash/college/pe1/miniproject/scripts/06_build_feature_matrix.py) implements a two-pass architecture:

1. **Pass 1 — Global Vocabulary Construction:**
   * Sequentially scans all k-mer dump text files to build a global frequency dictionary.
   * **Singleton Filtering:** K-mers that appear in only one genome are dropped (`MIN_GENOME_FREQ = 2`). Rare k-mers that exist in only a single isolate cannot provide generalizable predictive signal across a population and merely inflate dimensionality.
   * The remaining filtered k-mers are indexed into a vocabulary dictionary: $\text{kmer} \mapsto \text{column\_idx}$.

2. **Pass 2 — Coordinate (COO) to Compressed Sparse Row (CSR) Assembly:**
   * Reads each genome and appends non-zero elements to coordinate arrays: `(row_idx, col_idx, count)`.
   * The coordinate matrix is converted to a Compressed Sparse Row (`csr_matrix`) structure:
     * `data`: array of non-zero values (`uint16`).
     * `indices`: array of column indices.
     * `indptr`: array of row index pointers.
   * **Result:** The assembled matrix consumes only **~105 MB on disk** and ~300 MB in RAM, achieving a **98.9% memory reduction**.

---

## 6. Machine Learning Models: GBDTs vs Deep Learning

### Why Gradient Boosted Decision Trees (GBDT)?
Deep neural networks struggle on tabular k-mer matrices due to:
1. **Severe Extreme Dimensionality ($M \gg N$):** $2,132,407$ features vs $2,993$ training samples. Linear layers and dense projections drastically overfit without astronomical amounts of data.
2. **Extreme Sparsity:** Most bacteria possess only a small subset of the global k-mer dictionary.
3. **Interpretability Requirements:** Clinical healthcare models require direct biological feature attribution, which decision tree ensembles natively provide.

### Algorithms Implemented
* **LightGBM (Light Gradient Boosting Machine):**
  * Uses **Histogram-based algorithms** to bucket continuous feature counts into discrete bins (`max_bin=63`), dramatically speeding up split finding on sparse inputs.
  * Employs **Leaf-wise (best-first) tree growth**, splitting the leaf with maximum loss reduction.
  * Configured with `colsample_bytree = 0.3` to randomly sample 30% of k-mers at each iteration, preventing trees from relying solely on a single dominant marker.
* **XGBoost (eXtreme Gradient Boosting):**
  * Uses histogram-based exact split finding (`tree_method="hist"`).
  * Robust regularization parameters ($L_1$ and $L_2$ penalties) prevent leaf weight explosion.

---

## 7. Diagnostic Evaluation Metrics

In clinical microbiology, traditional **Accuracy** is an invalid metric:
* On an imbalanced dataset (e.g., 85% Susceptible, 15% Resistant), a dumb model predicting 100% "Susceptible" scores 85% accuracy.
* However, its clinical recall for resistant infections is **0%**—meaning every infected patient receives an ineffective drug, resulting in septic shock and mortality.

### Clinical Error Taxonomy
* **Very Major Error (VME) / False Susceptible:** The model predicts Susceptible when the isolate is Resistant. (Fatal risk).
* **Major Error (ME) / False Resistant:** The model predicts Resistant when the isolate is Susceptible. (Patient is denied first-line therapy and prescribed toxic alternatives).

### Evaluated Metrics
1. **Macro-Averaged F1-Score:**
   $$\text{Macro F1} = \frac{F1_{\text{Susceptible}} + F1_{\text{Resistant}}}{2}$$
   Treats both classes with equal importance regardless of sample count.
2. **Matthews Correlation Coefficient (MCC):**
   $$\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$
   Produces values in $[-1, +1]$, where $+1$ represents perfect prediction, $0$ represents random chance, and $-1$ represents complete inverse prediction.
3. **ROC-AUC & PR-AUC:** Area under Receiver Operating Characteristic and Precision-Recall curves to quantify discrimination independent of the decision threshold.
4. **Decision Threshold Optimization:**
   Instead of an arbitrary cutoff of $p \ge 0.5$, the pipeline sweeps thresholds $t \in [0.10, 0.90]$ to identify the operating point that maximizes the balance between sensitivity and specificity.

---

## 8. Biological Explainability: SHAP Game Theory & CARD Validation

### Shapley Additive Explanations (SHAP)
To eliminate the "black-box" nature of tree ensembles, the pipeline runs **SHAP TreeExplainer**:
* Based on cooperative game theory, Shapley values quantify the exact marginal contribution of each 31-mer feature towards shifting the model's prediction from the base expected value to the final log-odds output:
  $$f(x) = \phi_0 + \sum_{j=1}^{M} \phi_j(x)$$
* Features with strong positive $\phi_j$ values represent k-mers that promote resistance.
* Features with negative $\phi_j$ values represent k-mers that indicate susceptibility (e.g., wild-type intact outer membrane porins).

### Validation Against CARD
1. The top 20 most globally predictive 31-mers are exported to [results/top20_kmers.fasta](file:///home/yash/college/pe1/miniproject/results/top20_kmers.fasta).
2. These sequences can be searched directly using BLAST against the **Comprehensive Antibiotic Resistance Database (CARD)**:
   * **Expected Matches:** Sequences map directly to the active catalytic pocket of carbapenemases (*blaKPC-2*, *blaKPC-3*, *blaNDM-1*, *blaOXA-48*).
   * **Significance:** Proves that the algorithm identified genuine molecular resistance determinants through pure mathematical induction on raw sequence text, without human prior biological instruction.
