# Certified Robustness in Overparameterized Region

This repository contains the implementation and experiments for studying certified robustness in the overparameterized regime, comparing discriminative models (Ridge, AT) and generative models (Nearest Centroid, LDA).

## Project Structure

- `data/`: Data generation (synthetic) and loading (MNIST, CIFAR).
- `models/`: Implementation of Ridge, Nearest Centroid, and LDA.
- `utils/`: PGD Attack, AutoAttack, randomized smoothing, and plotting helpers.
- `config.py`: Global experiment parameters.
- `experiments.py`: Core logic for all experiments (A-E, MNIST, CIFAR).
- `main.py`: Main script to run the full pipeline.
- `requirements.txt`: Project dependencies.
- `Comp450_CertifiedRobustnessinOverparametrizedRegion.ipynb`: Interactive Jupyter Notebook.

## Installation

```bash
pip install -r requirements.txt
```

## Running Experiments

To run the full suite of experiments:
```bash
python main.py
```

Results and plots will be displayed, and logs will be saved to `experiment_logs_v2/`.
