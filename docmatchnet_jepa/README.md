# DocMatchNet-JEPA

DocMatchNet-JEPA is a research project for healthcare recommendation that compares:

1. **Embedding-space prediction (JEPA-style)**: learn predictive representation alignment between patient and provider views.
2. **Score-space prediction (original DocMatchNet style)**: directly optimize ranking/relevance scores.

The goal is to evaluate whether JEPA-style objectives provide better ranking quality, robustness, and sample efficiency than direct score prediction.

## Research Objectives

- Build a controlled experimental framework for healthcare recommendation.
- Compare **DocMatchNet-JEPA** against **DocMatchNet-Original** and standard baselines.
- Run ablations on gates, two-stage training, loss formulations, and gate dimensionality.
- Report ranking metrics, gate behavior analysis, and statistical significance tests.

## Project Layout

- `configs/`: default and experiment-specific configuration files.
- `src/models/`: JEPA model, original model, baselines, and shared components/losses.
- `src/data/`: dataset APIs, synthetic data generator, feature and embedding logic.
- `src/training/`: training loops for JEPA and original methods.
- `src/evaluation/`: ranking metrics, gate analysis, statistical tests, plots.
- `notebooks/`: experiment-driving notebooks for data generation, training, ablations, and figures.
- `results/`: tables, figures, checkpoints, and logs.
- `tests/`: unit tests for models, losses, and metrics.

## Configuration

Main config file: `configs/default_config.yaml`

It contains all major hyperparameters and controls, including:

- data generation and split strategy
- feature and embedding settings
- JEPA and original model hyperparameters
- loss composition (InfoNCE, VICReg, MSE, ranking)
- optimizer and scheduler settings
- training, checkpointing, and early stopping
- evaluation metrics and significance testing

Experiment overrides:

- `configs/docmatchnet_jepa.yaml`
- `configs/docmatchnet_original.yaml`
- `configs/ablation_configs/*.yaml`

## Setup

```bash
cd docmatchnet_jepa
pip install -r requirements.txt
pip install -e .
```

## Suggested Workflow

1. Generate or load datasets (`01_data_generation.ipynb`).
2. Train JEPA model (`02_train_docmatchnet_jepa.ipynb`).
3. Train original score-space model (`03_train_docmatchnet_original.ipynb`).
4. Train baseline models (`04_train_baselines.ipynb`).
5. Run ablations and sample-efficiency studies (`05`, `06`).
6. Analyze gates and generate final tables/figures (`07`, `08`, `09`).

## Planned Outputs

- Ranking quality: NDCG@K, MAP, MRR, Precision@K, Recall@K, HitRate@K, AUC
- Reliability: confidence intervals and statistical significance tests
- Interpretability: gate activation/importance analysis
- Artifacts: checkpoints, logs, publication-ready tables and figures

## Notes

This repository is structured for reproducible Kaggle or local experiments.
Implementation files are scaffolded and can be filled incrementally as experiments proceed.
