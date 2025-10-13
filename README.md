# Getting Started with CDGS (Compositional Diffusion with Guided Search)

This repository contains code and a Jupyter notebook used to explore CDGS-style compositional samplers for multimodal 2D synthetic datasets. The materials demonstrate experiments on compositional samplers with resampling/pruning heuristics.

## Quick Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Make sure you have uv installed, then:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment with dependencies
uv sync

# Activate the environment
source .venv/bin/activate
```
## Contents

- `getting_started_with_CDGS.ipynb` — Main notebook that builds models, defines CDGS samplers, runs experiments, and produces visualizations.
- `data.py` — Utilities to create synthetic 2D multimodal datasets and plotting helpers.
- `models.py` — Model definitions for diffusion and flow matching models.
- `samplers.py` — CDGS sampling implementations and related utilities.
- `utils.py` — General utility functions and helpers.

## Running the Notebook

After setting up the environment, start Jupyter and open the main notebook:

```bash
jupyter notebook getting_started_with_CDGS.ipynb
```

The notebook contains cells that will guide you through:
1. Dataset creation and visualization
2. Model training (or loading pretrained models)
3. CDGS sampling experiments
4. Results analysis and visualization

## Checkpoints and Reproducibility

Pretrained model checkpoints are provided under `pretrained_checkpoints/`
