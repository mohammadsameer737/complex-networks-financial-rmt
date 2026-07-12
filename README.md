# Financial Correlation Networks & Random Matrix Theory (RMT)

[![PyPI version](https://img.shields.io/badge/TestPyPI-v0.1.0-blue)](https://test.pypi.org/project/finnet-backbone/0.1.0/)

## 📦 Package Installation
The core backbone extraction and RMT modules are available as a standalone pip-installable package:

```bash
pip install -i https://test.pypi.org/simple/ finnet-backbone

## Project Overview
This project analyzes S&P 500 stock correlations using Random Matrix Theory (RMT) 
and extracts the multiscale network backbone using the Disparity Filter algorithm. 
It demonstrates how local heterogeneity in financial networks can be separated from 
global market noise to reveal underlying economic sector structures.

## Methodology
1. **Data Pipeline**: 10 years of S&P 500 daily adjusted close prices (2014-2024).
2. **Distance Metric**: Metric distance derived from Pearson correlation: $d_{ij} = \sqrt{2(1-\rho_{ij})}$.
3. **RMT Filtering**: Marchenko-Pastur distribution applied to the residual correlation matrix to separate market noise from systemic signal.
4. **Backbone Extraction**: Disparity Filter (Serrano et al., PNAS 2007) to extract the multiscale backbone, preserving local network heterogeneity.
5. **Robustness Analysis**: Simulated targeted attacks (highest degree) vs. random failures, tracking the Giant Connected Component (GCC) decay.

## Installation & Setup
Ensure you have Python 3.10+ installed. It is highly recommended to use a virtual environment.

```bash
# Create and activate virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt