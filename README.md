# Towards Fairer Hiring: A Modular Framework for Bias Mitigation and Explainability in Recruitment AI Systems

## Overview
This project implements a modular machine learning framework for evaluating bias mitigation strategies in recruitment AI systems.  
It enables controlled comparison of fairness, predictive performance, and explainability across multiple intervention stages.

The framework supports pre-processing, in-processing, and post-processing mitigation techniques, alongside SHAP-based explainability for analysing model behaviour.

---

## Research Context
AI-driven recruitment systems are increasingly used to support decision-making, but they risk reproducing historical biases present in training data.  
While many mitigation techniques exist, they are often evaluated in isolation, making it difficult to understand trade-offs between fairness and performance.

This project addresses this gap by providing a unified, modular framework for systematically evaluating mitigation strategies under consistent experimental conditions.

---

## Features
- Modular, configuration-driven pipeline  
- Baseline modelling (Logistic Regression, Random Forest, Gradient Boosting)  
- Pre-processing mitigation (Reweighing)  
- In-processing mitigation (Exponentiated Gradient - DP / EO constraints)  
- Post-processing mitigation (Threshold Optimisation)  
- SHAP-based explainability (global and subgroup analysis)  
- 5-fold stratified cross-validation  
- Multi-metric evaluation:
  - Performance: Accuracy, F1-score, ROC-AUC  
  - Fairness: Demographic Parity, Disparate Impact, Equalised Odds, Equal Opportunity  

---

## Project Structure

configs/ experiment configuration files
data/ raw datasets
notebooks/ exploratory analysis and visualisation
scripts/ execution scripts for each mitigation strategy
src/ core modules (data, models, mitigation, evaluation)
runs/ experimental outputs (CSV, JSON, plots)
requirements.txt dependencies


---
## Additional Scripts

The repository also contains an experimental script:

- `run_multistage_reweighing_expgrad.py`

This script explores a combined mitigation approach using both reweighing and exponentiated gradient.  
However, this approach was not included in the final evaluation due to instability and inconsistent results.

It is retained for completeness but is not part of the core experimental pipeline.


---
## Installation
Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Python 3.11 is recommended.

---

## Quick Start

```bash
python3 -m scripts.run_reweighing_cv
```

---

## Running Experiments

```bash
python3 -m scripts.run_reweighing_cv
python3 -m scripts.run_expgrad_cv
python3 -m scripts.run_threshold_optimizer_cv
```

Each script performs:
- data loading and preprocessing  
- model training  
- bias mitigation  
- cross-validation evaluation  
- results aggregation  

---

## Outputs

All outputs are saved in the `runs/` directory.

Each run contains:
- Cross-validation summaries (`cv_summary_*.csv`)  
- Fairness and performance metrics  
- Trade-off tables (`tradeoff_table.csv`, `tradeoff_deltas.csv`)  
- Visualisations (fairness-accuracy plots, SHAP plots)  

---

## Reproducibility

The framework follows a configuration-driven design.

Key experiment settings are defined in YAML files under `configs/`.

This ensures:
- reproducibility  
- consistency across experiments  
- controlled comparison of mitigation strategies  

---

## Author

Grace Omoike  
BSc Software Development