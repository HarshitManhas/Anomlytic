import os
import re
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from .patcher import extract_patches


class PatchDataset(Dataset):
    """PyTorch Dataset for time-series patches with their start indices."""

    def __init__(self, patches: np.ndarray, indices: np.ndarray):
        self.patches = torch.from_numpy(patches).float()
        self.indices = torch.from_numpy(indices).long()

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.patches[idx], self.indices[idx]


def load_tsb_ad_file(
    file_path: str | Path,
    fallback_train_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads a TSB-AD benchmark CSV file.
    Extracts the training split length from filename '_tr_<N>_'.
    Returns:
        (train_data, train_labels, test_data, test_labels)
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV file is empty: {path}")

    # All columns up to the last are features; last is Label
    data = df.iloc[:, :-1].values.astype(np.float32)
    labels = df.iloc[:, -1].values.astype(np.float32)

    # Parse train boundary from filename: e.g. '013_NAB_id_13_Traffic_tr_623_1st_2084.csv'
    match = re.search(r"_tr_(\d+)", path.name)
    if match:
        train_len = int(match.group(1))
    else:
        train_len = int(len(data) * fallback_train_ratio)

    train_data = data[:train_len]
    train_labels = labels[:train_len]
    test_data = data[train_len:]
    test_labels = labels[train_len:]

    return train_data, train_labels, test_data, test_labels


def download_tsb_ad(dataset_type: str = "TSB-AD-U", target_dir: str | Path = "data") -> Path:
    """
    Downloads and extracts TSB-AD benchmark dataset (TSB-AD-U or TSB-AD-M) from TheDatumOrg.
    """
    dataset_type = dataset_type.upper()
    if dataset_type not in ("TSB-AD-U", "TSB-AD-M"):
        raise ValueError("dataset_type must be either 'TSB-AD-U' or 'TSB-AD-M'")

    target_path = Path(target_dir) / dataset_type
    target_path.mkdir(parents=True, exist_ok=True)

    existing = list(target_path.glob("*.csv"))
    if len(existing) > 10:
        print(f"Dataset already exists with {len(existing)} files in {target_path}")
        return target_path

    url = f"https://www.thedatum.org/datasets/{dataset_type}.zip"
    zip_path = Path(target_dir) / f"{dataset_type}.zip"

    print(f"Downloading {dataset_type} from {url}...")
    urllib.request.urlretrieve(url, zip_path)

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    if zip_path.exists():
        zip_path.unlink()

    print(f"Successfully downloaded {dataset_type} to {target_path}")
    return target_path
