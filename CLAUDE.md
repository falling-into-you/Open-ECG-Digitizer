# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Open ECG Digitizer — a Python 3.12+ tool for digitizing 12-lead ECGs from scanned images or phone photos. Extracts raw time-series data (in microvolts) using a modular pipeline: U-Net segmentation, perspective correction, grid calibration, signal extraction, and lead identification.

## Commands

### Install
```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
git lfs pull  # download pre-trained weights (~110MB)
```

### Inference
```bash
python3 -m src.digitize --config src/config/inference_wrapper.yml
# Override config values:
python3 -m src.digitize --config src/config/inference_wrapper.yml DATA.output_path=my_output
```

### Training
```bash
python3 -m src.train  # uses src/config/unet.yml by default
```

### Evaluation
```bash
python3 -m src.evaluate --config src/config/evaluate.yml
```

### Format & Lint (all-in-one)
```bash
bash format_and_check.sh  # runs: black, isort, flake8, mypy
```

### Individual quality checks
```bash
black src
isort src
flake8 src
mypy src
```

### Tests
```bash
pytest                       # all tests
pytest test/test_config.py   # single file
pytest -v -s                 # verbose with stdout
```

CI enforces a 300-second timeout on tests.

## Code Style

- **black**: line-length 120
- **isort**: profile "black"
- **flake8**: max-line-length 120, ignores E203/E501/E704/W503/E226
- **mypy**: strict mode enabled; missing import stubs ignored for yacs, torchvision, scipy, scikit-image, transformers, sklearn.neighbors, torch_tps
- **Commits**: conventional commits enforced via gitlint (title max 70 chars, body max 72 chars/line)

## Architecture

### Entry Points

| Script | Purpose |
|--------|---------|
| `src/digitize.py` | CLI for batch inference on image folders |
| `src/train.py` | U-Net training with Ray Tune hyperparameter search |
| `src/evaluate.py` | Compute metrics (Pearson correlation) on digitized outputs |

### 8-Stage Inference Pipeline (`src/model/`)

All stages are orchestrated by `src/model/inference_wrapper.py`, which dynamically instantiates each module from YAML config:

1. **`unet.py`** — Semantic segmentation (RGB → 4-channel: grid, text/background, signal, background)
2. **`perspective_detector.py`** — Hough-transform-based perspective/rotation estimation from grid probability map
3. **`cropper.py`** — ROI extraction + perspective warp alignment using detected corners
4. **`pixel_size_finder.py`** — Autocorrelation-based grid spacing estimation (mm/pixel)
5. **`dewarper.py`** — Experimental TPS dewarping for curved paper (disabled by default)
6. **`signal_extractor.py`** — Connected-component analysis + Hungarian algorithm to extract trace lines
7. **`lead_identifier.py`** — Dual U-Net text detection + template matching against layout configs in `src/config/lead_layouts_*.yml`
8. Output: 12-lead canonical signals in µV, resampled to 5000 points

### Configuration System

Uses **yacs** (`CfgNode`) with dynamic class loading. Base template in `src/config/default.py`. All modules are specified by `class_path` strings and instantiated at runtime via `src/utils.import_class_from_path()`.

Key config files:
- `src/config/inference_wrapper.yml` — main inference config (model paths, device, resample size)
- `src/config/unet.yml` — training config (epochs, optimizer, loss, augmentation transforms)
- `src/config/lead_layouts_*.yml` — ECG lead layout templates for matching

Config values can be overridden from CLI: `KEY.SUBKEY=VALUE` (e.g., `DATA.output_path=foo MODEL.KWARGS.device=cpu`).

### Supporting Modules

| Directory | Purpose |
|-----------|---------|
| `src/dataset/` | `ECGScanDataset` (images + masks), `SyntheticLeadTextDataset` |
| `src/loss/loss.py` | `DiceFocalLoss` — hybrid dice + focal loss |
| `src/optimizer/adammuon.py` | `AdamMuon` — mixed Adam/Muon optimizer |
| `src/transform/vision.py` | 22 data augmentation transforms (crop, flip, contrast, zoom, QR overlay, JPEG artifacts, etc.) |
| `src/scripts/` | Dataset splitting, transform visualization, digital redaction |
| `src/report/` | Paper figure generation scripts |

### Pre-trained Weights

Stored via Git LFS in `weights/`:
- `unet_weights_07072025.pt` (87MB) — main segmentation model
- `lead_name_unet_weights_07072025.pt` (23MB) — lead text detector

### Tests

Two test files in `test/`:
- `test_config.py` — integration test running a 1-epoch training loop with test fixtures
- `test_utils.py` — unit test for `CosineToConstantLR` scheduler

Test fixtures: 3 ECG images + PNG segmentation masks in `test/test_data/`.

### CI/CD

GitHub Actions workflows in `.github/workflows/`:
- `test.yml` — pytest
- `lint.yml` — black + isort + flake8
- `type.yml` — mypy
- `commit_lint.yml` — conventional commit enforcement
- `release.yml` / `release_noop.yml` — semantic-release version bumping
