"""
Anomlytic: Fast & Lightweight 1D CNN + Patch Representation Learning for Time-Series Anomaly Detection.
"""

from .model import PatchCNN1D
from .detector import AnomlyticDetector
from .dataset import load_tsb_ad_file, download_tsb_ad, PatchDataset
from .patcher import extract_patches, distribute_scores_to_points
from .metric_learning import TemporalTripletLoss, TemporalContrastiveLoss

__version__ = "0.1.0"
__all__ = [
    "PatchCNN1D",
    "AnomlyticDetector",
    "load_tsb_ad_file",
    "download_tsb_ad",
    "PatchDataset",
    "extract_patches",
    "distribute_scores_to_points",
    "TemporalTripletLoss",
    "TemporalContrastiveLoss",
]
