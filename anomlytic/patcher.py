import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view


def extract_patches(
    data: np.ndarray,
    patch_size: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts sliding window patches from univariate (T,) or multivariate (T, C) array.
    Returns:
        patches: float32 array shaped (N, C, patch_size)
        indices: int64 array of starting time-indices shaped (N,)
    """
    if data.ndim == 1:
        data = data[:, np.newaxis]  # (T, 1)

    T, C = data.shape
    if T < patch_size:
        raise ValueError(f"Series length {T} is shorter than patch_size {patch_size}")

    # sliding_window_view on axis 0: shape (num_windows, C, patch_size)
    windows = sliding_window_view(data, window_shape=patch_size, axis=0)[::stride]
    # windows shape: ((T - patch_size) // stride + 1, C, patch_size)
    # Transpose if needed: window_shape creates extra dim at end: (num_windows, C, patch_size)
    patches = np.ascontiguousarray(windows, dtype=np.float32)
    indices = np.arange(0, T - patch_size + 1, stride, dtype=np.int64)

    return patches, indices


def distribute_scores_to_points(
    patch_scores: np.ndarray,
    patch_size: int,
    num_points: int,
    stride: int = 1,
) -> np.ndarray:
    """
    Distributes patch-level anomaly scores back to point-level scores via rolling average.
    Returns:
        point_scores: 1D float32 array of shape (num_points,)
    """
    scores = np.nan_to_num(patch_scores.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)

    # Upsample if stride > 1
    if stride > 1:
        expanded = np.zeros(num_points - patch_size + 1, dtype=np.float32)
        idx = np.arange(len(scores)) * stride
        valid = idx < len(expanded)
        expanded[idx[valid]] = scores[valid]
        scores = expanded

    kernel = np.ones(patch_size, dtype=np.float32)
    sums = np.convolve(scores, kernel, mode="full")[:num_points]
    counts = np.convolve(np.ones_like(scores), kernel, mode="full")[:num_points]

    point_scores = np.divide(
        sums, counts, out=np.zeros(num_points, dtype=np.float32), where=counts > 0
    )
    return np.nan_to_num(point_scores, nan=0.0, posinf=1.0, neginf=0.0)
