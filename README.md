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

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

## 🚀 Execution Modes

This project offers two execution modes depending on the depth of analysis required:

### 1. Core Workflow (Fast - ~5 mins)
Runs the essential data pipeline, RMT filtering, backbone extraction, and robustness analysis. 
Ideal for CI/CD testing and quick validation.
```bash
python pipeline.py