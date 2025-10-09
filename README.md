# Getting Started with CDGS (Compositional Diffusion with Guided Search)

This repository contains code and a Jupyter notebook used to explore CDGS-style compositional samplers for multimodal 2D synthetic datasets. The materials demonstrate experiments on compositional samplers with resampling/pruning heuristics.

Contents
- `getting_started_with_CDGS.ipynb` — Main notebook that builds models, defines CDGS samplers, runs experiments, and produces visualizations.
- `dataset.py` — Utilities to create synthetic 2D multimodal datasets and plotting helpers.
- `plotter.py` — Standalone plotting utilities and helpers to visualize latents and Gaussian overlays.
- `pretrained_checkpoints/` — Pretrained model weights (not tracked by git in default `.gitignore`).

Quick setup: Create and activate a Python environment (recommended Python 3.10+).

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

and just follow the notebook.

- The notebook contains an install cell that uses `%pip` so run that first to ensure kernel dependencies are available in the active environment.

- The notebook will save results as csv files which can be converted into 
images using `plotter.py`. Example:

```bash
python plotter.py --csv /path/to/results.csv
```

Checkpoints and reproducibility
- Pretrained model checkpoints are provided under `pretrained_checkpoints/`
- The notebook includes a `seed_everything` helper and device selection that will attempt CUDA/MPS and fall back to CPU.
