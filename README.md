# Financial Correlation Networks & Random Matrix Theory

[![PyPI version](https://img.shields.io/badge/TestPyPI-v0.1.0-blue)](https://test.pypi.org/project/finnet-backbone/0.1.0/)
[![CI](https://github.com/mohammadsameer737/complex-networks-financial-rmt/actions/workflows/pytest.yml/badge.svg)](https://github.com/mohammadsameer737/complex-networks-financial-rmt/actions)

## Overview

This project analyzes S&P 500 stock correlations using Random Matrix Theory (RMT) 
and extracts the multiscale network backbone using the Disparity Filter algorithm. 
It demonstrates how local heterogeneity in financial networks can be separated from 
global market noise to reveal underlying economic sector structures.

## Methodology

1. **Data Pipeline**: 10 years of S&P 500 daily adjusted close prices (2014-2024).
2. **Distance Metric**: Metric distance derived from Pearson correlation: $d_{ij} = \sqrt{2(1-\rho_{ij})}$.
3. **RMT Filtering**: Marchenko-Pastur distribution applied to the residual correlation matrix to separate market noise from systemic signal.
4. **Backbone Extraction**: Disparity Filter (Serrano et al., PNAS 2009) to extract the multiscale backbone, preserving local network heterogeneity.
5. **Robustness Analysis**: Simulated targeted attacks (highest degree) vs. random failures, tracking the Giant Connected Component (GCC) decay.

## Installation

Ensure you have Python 3.10+ installed. It is highly recommended to use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```
## 🚀 Execution Modes

This project offers two execution modes depending on the depth of analysis required:

### 1. Core Workflow (Fast - ~5 mins)
Runs the essential data pipeline, RMT filtering, backbone extraction, and robustness analysis. 
Ideal for CI/CD testing and quick validation.

```bash
python pipeline.py
```

### 2. Full Comprehensive Analysis (Slow - ~1 hour)
Runs the complete 10-year temporal analysis, parameter sweeps, and ensemble simulations. 
This generates the full suite of 11 figures used in the technical report.

```bash
python -m src.main
```

## ️ Architectural Decisions

This project implements several deliberate design choices to prioritize cross-platform reproducibility and end-user accessibility, aligning with modern software engineering best practices:

1. **Pure-Python Workflow Orchestrator**: Instead of Snakemake (which requires C-extension compilation on Windows, causing platform-specific failures), this project uses `pipeline.py`. This pure-Python orchestrator ensures 100% cross-platform compatibility without requiring students to install Visual Studio Build Tools.
2. **Vectorized NumPy Implementation**: Instead of multi-language integration (e.g., Python + C/C++), this project achieves comparable high-performance computing results through NumPy vectorization (demonstrating an 18x speedup in `benchmarks/performance_comparison.py`), maintaining code maintainability and accessibility.
3. **CSV-based Data Management**: For this financial network analysis scope, local CSV caching provides sufficient data persistence and manipulation. The modular architecture is designed to support seamless database integration (e.g., SQLite/PostgreSQL) for future scaling if required.

## 📄 Documentation

- **[System Architecture Report](REPORT.md)**: Engineering, pipeline design, and software lifecycle details.
- **[Mathematical Walkthrough](01_RMT_and_Disparity_Math.ipynb)**: Jupyter notebook mapping theoretical equations directly to code implementation.
- **[Full Technical Report (PDF)](docs/complex_networks_report.pdf)**: Complete academic derivations, proofs (e.g., triangle inequality), and physical discussion.

## 📦 Open Source Contribution

The core backbone extraction and RMT modules have been packaged and published to the Python Package Index (TestPyPI) to demonstrate full software lifecycle management:

```bash
pip install -i https://test.pypi.org/simple/ finnet-backbone
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
