# AI Independent Study · 2026–2027

This repository contains the coursework, experiments, research projects,
manuscripts, and reproducibility records for my AI Independent Study during
the 2026–2027 academic year.

The work is organized by quarter. Each quarter keeps its code, configurations,
notebooks, raw results, figures, and written manuscript together, along with
a log of any outside or AI assistance used. The goal is that every reported
result can be traced back to the exact code, configuration, data split, and
environment that produced it.

| | |
|---|---|
| **Student** | _TBD_ |
| **Advisor** | _TBD_ |
| **Institution** | _TBD_ |
| **Academic year** | 2026–2027 |

---

## Repository structure

```
ai-independent-study-2026-2027/
├── README.md                  ← you are here
├── requirements.txt           ← pinned packages from the working environment
├── .gitignore
├── environment/
│   ├── README.md              ← environment setup and verification guide
│   ├── system_info.txt        ← recorded OS, Python, PyTorch, and device info
│   └── verify_environment.py  ← re-runs all environment checks
├── Q1/
│   ├── README.md              ← quarter overview, goals, weekly log
│   ├── assistance_log.md      ← record of outside / AI assistance
│   ├── configs/               ← experiment configuration files
│   ├── data/
│   │   ├── README.md
│   │   └── split_indices/     ← tracked train/val/test indices (datasets are not)
│   ├── notebooks/
│   │   └── week01_fashion_mnist_baseline.ipynb
│   ├── src/                   ← reusable Python modules
│   ├── results/raw/           ← unprocessed metrics and logs from runs
│   ├── figures/               ← plots generated from results
│   └── manuscript/            ← write-ups and papers
├── Q2/  (same layout as Q1, without data/)
├── Q3/
└── Q4/
```

### What goes where

| Folder | Contents | Tracked by Git? |
|---|---|---|
| `configs/` | Hyperparameters, seeds, and run settings (YAML/JSON) | Yes |
| `notebooks/` | Exploratory and assignment notebooks, named `weekNN_topic.ipynb` | Yes |
| `src/` | Reusable code imported by notebooks and scripts | Yes |
| `results/raw/` | Metrics, logs, and CSV/JSON outputs straight from a run | Yes (keep files small) |
| `figures/` | Plots produced from `results/raw/` | Yes |
| `manuscript/` | Reports, papers, drafts | Yes |
| `data/split_indices/` | Index files that define dataset splits | Yes |
| `data/*` (datasets) | Downloaded datasets | **No** |
| Checkpoints (`*.pt`, `*.pth`, `*.ckpt`, `checkpoints/`) | Trained model weights | **No** |

---

## Getting started

Requires macOS on Apple silicon and Python 3.14 (the environment was verified
with Python 3.14.0 and PyTorch 2.14.0; see `environment/system_info.txt`).

```bash
# 1. Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install the pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify the environment (writes environment/system_info.txt)
python environment/verify_environment.py

# 4. Launch JupyterLab
jupyter lab
```

In notebooks, select the device with MPS first and CPU as the fallback:

```python
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

---

## Reproducibility practices

1. **Fix seeds.** Set seeds for `random`, `numpy`, and `torch` at the top of every
   notebook or script, and record them in the run's config.
2. **Save splits.** Write train/validation/test indices to `data/split_indices/`
   instead of relying on re-shuffling.
3. **Config per run.** Store the settings for each reported experiment in
   `configs/`.
4. **Keep raw results.** Save unedited metrics to `results/raw/`; generate figures
   from those files, not from memory.
5. **Record the environment.** Re-run `environment/verify_environment.py` and
   regenerate `requirements.txt` (`pip freeze > requirements.txt`) whenever
   packages change.
6. **Disclose assistance.** Log any help from people, tutorials, or AI tools in
   that quarter's `assistance_log.md`.

---

## Quarter overview

| Quarter | Focus | Status |
|---|---|---|
| [Q1](Q1/README.md) | Deep learning foundations: starting with a Fashion-MNIST baseline | In progress |
| [Q2](Q2/README.md) | _TBD_ | Not started |
| [Q3](Q3/README.md) | _TBD_ | Not started |
| [Q4](Q4/README.md) | _TBD_ | Not started |
