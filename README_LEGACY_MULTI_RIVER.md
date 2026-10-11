> Archived six-river overview from GitHub commit `f34175a` (2026-10-02). The website instructions and screenshot below describe the retired version. For the current two-river English website and Cloudflare Pages setup, see [README.md](README.md).

<div align="center">

# 🌊 The Yellow River

### Six rivers. One question: what can river models learn from each other?

A research portfolio exploring river forecasting, physics-guided learning,<br>
and knowledge transfer between the Yellow River and the Mississippi.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/Models-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Rivers](https://img.shields.io/badge/Rivers-6-087E8B)
![Research Demo](https://img.shields.io/badge/Data-Synthetic-E9C46A)

[Overview](#overview) · [Quick Start](#quick-start) · [Results](#results) · [Transfer Learning](#transfer-learning) · [Website](#website) · [Documentation](#documentation)

</div>

[Archived River Model Lab website preview](https://github.com/Wiiiiiiiley/The-Yellow-River/blob/f34175adb79139306449e72793fc36291299438b/website/assets/preview.jpg)

> **Research demonstration, not an operational forecasting system.**<br>
> The included multi-river datasets are synthetic. Results describe performance on those datasets—not validated forecasting accuracy on real rivers.

## Overview

This project began with Yellow River modeling and grew into a comparative river-learning laboratory.

It brings together six river profiles, reproducible forecasting experiments, a bidirectional transfer-learning study, and an interactive website designed for a personal research portfolio.

| River | Identifier |
|:------|:-----------|
| Yellow River | `yellow_river` |
| Mississippi | `mississippi` |
| Yangtze | `yangtze` |
| Amazon | `amazon` |
| Nile | `nile` |
| Danube | `danube` |

Each profile contains five illustrative river reaches. These are modeling nodes, not calibrated gauging stations.

### At a glance

| Component | Included |
|:----------|:---------|
| Input window | Past 24 hours |
| Forecast horizon | Next 6 hours |
| Prediction targets | Water depth, discharge, sediment concentration |
| Main experiments | Physics-guided model, data-only model, persistence baseline |
| Transfer study | Yellow River ↔ Mississippi, three random seeds |
| Saved checkpoints | 12 main models + 18 transfer models |
| Presentation | Responsive, dependency-free static website |

## Quick Start

### Explore the website

No model training or JavaScript build step is required.

```bash
git clone --branch feature/enhanced-pinn-v3.0 \
  https://github.com/Wiiiiiiiley/The-Yellow-River.git

cd The-Yellow-River
python3 -m http.server 8080
```

Open **http://localhost:8080/website/**.

The current website interface and supporting study documents are primarily in Chinese; this README provides an English project overview.

### Set up the modeling environment

The multi-river pipeline requires Python 3.10+, NumPy, and PyTorch.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy torch
```

The broader dependencies in `requirements.txt` support the original Yellow River workflow and additional tools.

## Modeling Approach

The multi-river model combines **directed upstream aggregation** with a shared **GRU temporal encoder**.

```text
24 hours of observations
          ↓
Directed river-graph aggregation
          ↓
GRU temporal encoder
          ↓
6-hour forecasts across five reaches
```

Inputs include water depth, discharge, sediment concentration, temperature, rainfall, and a seasonal feature.

Three methods are evaluated:

| Method | Purpose |
|:-------|:--------|
| Physics-guided graph GRU | Adds a Manning-based depth–discharge consistency penalty and nonnegativity penalties |
| Data-only graph GRU | Uses the same architecture without physics regularization |
| Persistence | Repeats the latest observed value as a simple baseline |

The physics-guided model is **not a full PDE-constrained PINN** and does not solve the complete Saint-Venant or Exner equations. The repository also retains the original Yellow River PINN/STGNN experiments as a separate workflow.

### Reproduce the experiments

```bash
python3 scripts/run_multi_river.py \
  --epochs 25 \
  --hours 8760 \
  --seed 42 \
  --transfer-seeds 42,43,44
```

This regenerates experiment outputs, including checkpoints, predictions, metrics, and website data.

The evaluation uses chronological 70% / 15% / 15% splits. Normalization statistics come from the training period, and forecast targets do not cross split boundaries.

## Results

The included snapshot uses **8,760 synthetic hourly samples per river**.

### Discharge forecasting

Test RMSE in **m³/s**; lower is better.

| River | Physics-guided | Data-only | Persistence |
|:------|---------------:|----------:|------------:|
| Yellow River | 9.70 | 9.70 | 10.28 |
| Mississippi | 86.73 | 86.71 | 88.86 |
| Yangtze | 142.18 | 142.26 | 144.27 |
| Amazon | 607.09 | 607.07 | 611.63 |
| Nile | 9.59 | 9.59 | 9.88 |
| Danube | 22.84 | 22.84 | 23.94 |

**What this shows:** the learned models modestly outperform persistence in this synthetic snapshot. Differences between the physics-guided and data-only models are small; these results do not establish a consistent benefit from physics regularization.

Absolute errors should not be used to rank rivers against each other because their discharge scales differ.

Additional metrics include MAE, bias, NSE, and range-normalized RMSE. Forecast bands use empirical validation residuals, with coverage evaluated on the test set; they are not guaranteed confidence intervals.

## Transfer Learning

Can a model trained on one river help forecast another?

The study compares three strategies in both directions:

- **Target-only:** train from scratch using a reduced target-label budget.
- **Zero-shot:** apply the source-trained model without target fine-tuning.
- **Fine-tuned:** adapt the source-trained model using the same reduced target-label budget.

### Included findings

Mean NRMSE across water depth, discharge, and sediment, averaged over three seeds:

| Transfer direction | Target-only | Zero-shot | Fine-tuned | Error change vs. target-only |
|:-------------------|------------:|----------:|-----------:|----------------------------:|
| Yellow River → Mississippi | 0.0226 | 0.0493 | 0.0246 | **+8.9%** |
| Mississippi → Yellow River | 0.0193 | 0.0341 | 0.0182 | **−5.6%** |

**Transfer is asymmetric.** Fine-tuning helps in the Mississippi-to-Yellow direction but increases error in the reverse direction.

These experiments demonstrate both positive and negative transfer—not a universal advantage.

> Target training-input statistics are available for normalization, including in the zero-shot setting. Target validation labels are used for model selection. Source pretraining also adds computational cost.

## Website

The static showcase makes the experiments accessible through:

- Six river profiles and interactive forecasts.
- Water-depth, discharge, and sediment views.
- Five-reach selection and backtest comparisons.
- Prediction bands, model metrics, and transfer results.
- Source references and downloadable artifacts.

### Preview the publication bundle

```bash
python3 -m http.server 8080 --directory public_river_lab
```

Open **http://localhost:8080/**.

To export a fresh bundle into a new directory:

```bash
python3 scripts/export_river_site.py --output public_river_lab_new
```

The exporter packages existing results; it does not retrain models and refuses to overwrite an existing destination.

A ready-to-upload archive is included: [public_river_lab.zip](public_river_lab.zip). No public deployment URL is configured in this repository.

## Working with Real Data

The pipeline accepts explicit observed-data inputs:

```bash
python3 scripts/run_multi_river.py --data-dir /path/to/river_csvs
```

Provide a `<river_id>.csv` file for every selected river, using these columns:

```text
timestamp,node_id,water_depth,discharge,sediment,temperature,rainfall
```

Expected units:

| Variable | Unit |
|:---------|:-----|
| Water depth | m |
| Discharge | m³/s |
| Sediment concentration | kg/m³ |
| Temperature | °C |
| Rainfall | mm/h |

Real-data use requires consistent timestamps, reach mapping, quality control, and calibrated physical parameters. The included illustrative geometry should not be treated as measured river geometry.

## Project Structure

```text
The-Yellow-River/
├── configs/                   # River profiles and source references
├── src/                       # Models and supporting code
├── scripts/                   # Training, prediction, export, verification
├── experiments/multi_river/   # Saved models, forecasts, metrics, manifests
├── docs/                      # Study notes and publication guidance
├── tests/                     # Automated tests
├── website/                   # Interactive research showcase
├── public_river_lab/          # Self-contained publication bundle
└── public_river_lab.zip       # Packaged static-site export
```

## Verification

```bash
python3 -B -m unittest discover -s tests -v
python3 -B scripts/verify_multi_river.py
```

Verification covers saved artifacts, checkpoint loading, forecast reproduction, training-only normalization, and publication downloads.

## Documentation

The following detailed guides are currently written in Chinese:

| Guide | Topic |
|:------|:------|
| [Multi-river study](docs/multi_river_study.md) | Experimental design and modeling choices |
| [Yellow River–Mississippi comparison](docs/yellow_mississippi_comparison.md) | River comparison and transfer-learning methodology |
| [Results report](docs/multi_river_results.md) | Generated metrics and interpretation |
| [Publishing guide](docs/publishing_guide.md) | Website packaging and deployment preparation |
| [Original training guide](README_TRAINING.md) | Legacy Yellow River workflow |
| [Real-data guide](README_REAL_DATA.md) | Original observed-data workflow |

Background references are recorded in [configs/river_sources.json](configs/river_sources.json). They support the scientific context, not validation of the synthetic data or illustrative parameters.

---

<div align="center">

**Understand the river. Test the model. Show the evidence.**

A personal research portfolio connecting hydrology, machine learning, and transparent experimentation.

</div>
