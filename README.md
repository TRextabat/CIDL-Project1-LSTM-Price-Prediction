# CIDL Project 1 -- Fuzzy Logic, Swarm Intelligence and Deep Learning

**Student:** Amir Amiri Tabat (05220000102)
**Course:** Computational Intelligence and Deep Learning (CIDL)
**University:** Ege University, Faculty of Engineering, Computer Engineering
**Semester:** 2025--2026 Spring

---

## Project Overview

This project covers 6 sections as required by the CIDL course:

1. **Fuzzy Logic & Swarm Intelligence** -- Fuzzy set theory, scikit-fuzzy application, ABC vs DE comparison
2. **ML / DL / Gen AI** -- ROC curves, Transformer architecture, ensemble methods, GenAI ethics
3. **Deep Learning Application (LSTM)** -- Production-level LSTM price prediction on XAUUSD and BTCUSD
4. **Evaluator & Auditor Roles** -- Performance assessment, compliance, ethical AI
5. **Project Participants** -- Solo submission with task allocation
6. **Self-Assessment** -- Grading table with explanations

---

## Final Report

The final report is located at:

```
CIDL_Project1_Report.pdf      <-- Main report (59 pages)
```

---

## Directory Structure

```
CIDL_project1/
|
|-- CIDL_Project1_Report.pdf          # Final report (PDF)
|-- CIDL_Project1_Report.md           # Report source (Markdown)
|-- README.md                         # This file
|-- pyproject.toml                    # Python project config & dependencies
|-- CIDL_2025_P1_v3.doc              # Original assignment (professor's doc)
|-- CIDL_2025_P1_v3.md               # Assignment converted to text
|
|-- configs/
|   `-- experiment.yaml               # Experiment configuration (models, assets, hyperparams)
|
|-- src/cidl/                         # Main source code
|   |-- data/
|   |   |-- loader.py                 # Load OHLCV parquet data from FinAgent
|   |   |-- features.py               # Compute 20 technical indicators (pandas-ta)
|   |   |-- scaler.py                 # RobustScaler (walk-forward safe)
|   |   `-- dataset.py                # PyTorch TimeSeriesDataset
|   |
|   |-- models/
|   |   |-- base.py                   # Abstract base model class
|   |   |-- simple_lstm.py            # 1-layer LSTM (76K params, baseline)
|   |   |-- stacked_lstm.py           # 3-layer deep LSTM (357K params)
|   |   |-- bilstm.py                 # Bidirectional LSTM (581K params)
|   |   |-- lstm_attention.py         # LSTM + Bahdanau Attention (233K params)
|   |   `-- registry.py               # Model name -> class mapping
|   |
|   |-- training/
|   |   |-- trainer.py                # Training loop (Adam, early stop, grad clip, GPU)
|   |   |-- walk_forward.py           # Walk-forward cross-validation (3 folds + purge)
|   |   `-- metrics.py                # Regression, classification, financial, statistical metrics
|   |
|   |-- evaluation/
|   |   `-- evaluator.py              # Aggregate results, stat tests (ADF, DM, PT)
|   |
|   |-- visualization/
|   |   |-- plots.py                  # All figure generation (matplotlib/seaborn)
|   |   `-- report.py                 # PDF report generator (fpdf2)
|   |
|   `-- utils/
|       |-- device.py                 # GPU/CUDA detection
|       `-- seed.py                   # Reproducibility (seed=42)
|
|-- scripts/
|   |-- run_experiments.py            # Main entry: runs all 32 experiments on GPU
|   |-- generate_report.py            # Generate Section 3 PDF report
|   |-- generate_full_report.py       # Generate complete 6-section PDF (fpdf2)
|   |-- fuzzy_student_evaluation.py   # Section 1.iv: Fuzzy logic application (scikit-fuzzy)
|   |-- generate_diagrams.py          # Architecture diagrams (matplotlib)
|   |-- create_composite_figures.py   # Composite grid figures (4x4 grids)
|   `-- mermaid_diagrams.sh           # Professional Mermaid architecture diagrams
|
|-- outputs/
|   |-- results/                      # 32 experiment JSON files + summary.json
|   |-- figures/                      # 148+ individual figures + 13 composites + 6 mermaid
|   |-- models/                       # Saved model checkpoints
|   `-- report/                       # Generated reports (PDF, Markdown)
|
`-- .venv/                            # Python virtual environment (not in git)
```

---

## How to Run

### Prerequisites
- Python 3.11+
- NVIDIA GPU with CUDA support (RTX 3050 Ti or better)
- Node.js (for Mermaid diagrams)

### Setup
```bash
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install pandas numpy pyarrow pandas-ta scikit-learn statsmodels arch optuna matplotlib seaborn fpdf2 tqdm pyyaml scipy scikit-fuzzy
```

### Run Experiments (32 experiments on GPU, ~27 min)
```bash
source .venv/bin/activate
python scripts/run_experiments.py
```

### Generate Figures
```bash
python scripts/fuzzy_student_evaluation.py     # Fuzzy logic figures
python scripts/generate_diagrams.py            # Architecture diagrams
python scripts/create_composite_figures.py     # Composite grids
bash scripts/mermaid_diagrams.sh               # Mermaid diagrams
```

### Generate PDF Report
```bash
pandoc outputs/report/CIDL_Project1_Report.md -o CIDL_Project1_Report.pdf --pdf-engine=xelatex --toc
```

---

## Key Results

| Model | Asset | Best Sharpe | Dir Accuracy | RMSE |
|-------|-------|-------------|--------------|------|
| BiLSTM | BTCUSD 1D | 0.794 | 51.6% | 0.035 |
| LSTM+Attention | XAUUSD 1H | 0.307 | 51.6% | 0.002 |
| Stacked LSTM | XAUUSD 1H | 0.291 | 51.5% | 0.002 |
| Simple LSTM | BTCUSD 1D | 0.719 | 51.6% | 0.039 |

---

## Data Source

Training data sourced from the [FinAgent](../FinAgent/) algorithmic trading system:
- `xauusd_1h.parquet` -- Gold hourly (33,533 bars, 2020-2026)
- `btcusd_1d.parquet` -- Bitcoin daily (4,056 bars, 2015-2026)

---

## Technologies

PyTorch 2.6 | pandas-ta | scikit-learn | statsmodels | matplotlib | seaborn | fpdf2 | optuna | Mermaid | pandoc/xelatex
