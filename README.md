# Anomlytic

> **Fast, Lightweight 1D CNN + Patch Representation Learning for Time-Series Anomaly Detection**

Anomlytic is a clean, self-contained Python / PyTorch implementation of patch-based representation learning for time-series anomaly detection. Instead of heavy, slow sequence models or complex transformers, Anomlytic extracts temporal patches, learns rich representations using a 1D CNN with metric learning, and scores anomalies using a compact prototype memory bank.

---

## Architecture Overview

```
[Time Series] ──► [Sliding Patches (L, stride)] ──► [1D CNN + Stats] ──► [Metric Learning (h, z)]
                                                                               │
                                                   ┌───────────────────────────┘
                                                   ▼
[Test Series] ──► [Test Patches] ──► [Cosine Dist to Memory Bank] ──► [1D Conv Smoothing] ──► Point-wise Scores
```

1. **Patch Extraction & Normalization**: Sliding temporal window ($L=64$, stride $s=1$). Zero-copy sliding view with scale-robust normalization.
2. **1D CNN Backbone**: 4-layer 1D CNN (kernel sizes 7, 5, 3, 3) + global adaptive average pooling. Local shape dynamics are coupled with patch-level statistics (mean, variance) to detect both pattern distortions and amplitude shifts.
3. **Metric Learning Objective**:
   - `TemporalTripletLoss`: Pulls temporal neighbors within radius $r$ close while repelling hard negative patches.
   - `TemporalContrastiveLoss`: Multi-negative InfoNCE contrastive objective with temporal neighbor positive mining.
4. **Memory Bank Coreset**: Fits `MiniBatchKMeans` on normal training embeddings to extract compact prototype centroids, guaranteeing sub-second inference.
5. **Score Redistribution**: 1D convolution rolling average smoothly maps patch-level anomaly distances back to point-level anomaly scores.

---

## Installation

```bash
git clone <your-repo-url>
cd Anomlytic
pip install -r requirements.txt
```

---

## Datasets (TSB-AD)

Anomlytic directly supports the [TSB-AD benchmark](https://github.com/TheDatumOrg/TSB-AD) (`TSB-AD-U` univariate and `TSB-AD-M` multivariate). Sample files are included in `data/`:
- `data/TSB-AD-U/`: Univariate benchmark samples (NAB Traffic, UCR Medical)
- `data/TSB-AD-M/`: Multivariate benchmark samples (MSL Sensor, SMAP Sensor)

To download the full TSB-AD datasets automatically:
```bash
# Download and extract full TSB-AD-U (1000+ univariate series)
python cli.py --download TSB-AD-U

# Download and extract full TSB-AD-M (multivariate series)
python cli.py --download TSB-AD-M
```

---

## Quick Start (CLI)

### 1. Run on Univariate Benchmark
```bash
python cli.py --data data/TSB-AD-U --patch_size 64 --iters 50
```

### 2. Run on Multivariate Benchmark
```bash
python cli.py --data data/TSB-AD-M --patch_size 64 --iters 50
```

### CLI Options
| Argument | Default | Description |
|---|---|---|
| `--data` | `data/TSB-AD-U` | Path to CSV file or directory of CSVs |
| `--output` | `results` | Output directory for scores and summary metrics |
| `--download` | `None` | `TSB-AD-U` or `TSB-AD-M` (auto downloads from TheDatumOrg) |
| `--patch_size` | `64` | Patch length $L$ |
| `--stride` | `1` | Sliding window stride $s$ |
| `--loss` | `triplet` | Metric learning objective (`triplet` or `contrastive`) |
| `--iters` | `100` | Number of metric learning training iterations |
| `--lr` | `1e-3` | Learning rate with cosine decay |
| `--batch_size` | `256` | Batch size |
| `--top_k` | `3` | Number of nearest memory prototypes for anomaly score |
| `--device` | `auto` | `cpu` or `cuda` |

---

## Python API Usage

```python
import numpy as np
from anomlytic import AnomlyticDetector, load_tsb_ad_file

# 1. Load data from TSB-AD format
train_x, train_y, test_x, test_y = load_tsb_ad_file("data/TSB-AD-U/013_NAB_id_13_Traffic_tr_623_1st_2084.csv")
full_x = np.concatenate([train_x, test_x], axis=0)
full_y = np.concatenate([train_y, test_y], axis=0)

# 2. Fit detector on normal train segment
detector = AnomlyticDetector(patch_size=64, num_iters=50, lr=1e-3)
detector.fit(train_x)

# 3. Predict point-wise anomaly scores
scores = detector.predict_score(full_x)

# 4. Evaluate detection performance
metrics = detector.evaluate(scores, full_y)
print("AUC-ROC:", metrics["AUC-ROC"])
print("AUC-PR: ", metrics["AUC-PR"])
print("Best-F1:", metrics["Best-F1"])
```

---

## Running Tests

Run the self-check verification test suite:
```bash
python tests/test_anomlytic.py
```
Tests cover:
- Sliding-window patch extraction and rolling convolution score redistribution
- 1D CNN forward pass & projection head normalization
- Triplet and contrastive metric learning losses
- Synthetic anomaly detection benchmark (asserts AUC-ROC > 0.85)
- Real TSB-AD sample benchmark evaluation
