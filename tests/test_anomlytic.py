import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from anomlytic.model import PatchCNN1D
from anomlytic.patcher import extract_patches, distribute_scores_to_points
from anomlytic.metric_learning import TemporalTripletLoss, TemporalContrastiveLoss
from anomlytic.detector import AnomlyticDetector
from anomlytic.dataset import load_tsb_ad_file


def test_patch_extraction_and_distribution():
    # Synthetic series of length 100
    data = np.sin(np.linspace(0, 10, 100))
    patch_size = 16
    patches, indices = extract_patches(data, patch_size=patch_size, stride=1)
    assert patches.shape == (85, 1, 16), f"Unexpected shape {patches.shape}"
    assert len(indices) == 85

    # Test distribution back to points
    scores = np.ones(85, dtype=np.float32)
    point_scores = distribute_scores_to_points(scores, patch_size=patch_size, num_points=100)
    assert len(point_scores) == 100
    assert np.allclose(point_scores, 1.0)
    print("[PASS] Patch extraction & distribution test passed.")


def test_patch_cnn_forward():
    # Univariate batch: (8, 1, 32)
    x_uni = torch.randn(8, 1, 32)
    model_uni = PatchCNN1D(in_channels=1, embed_dim=64, projection_dim=128)
    h_uni, z_uni = model_uni(x_uni)
    assert h_uni.shape == (8, 64)
    assert z_uni.shape == (8, 128)
    assert torch.allclose(torch.norm(z_uni, dim=-1), torch.ones(8), atol=1e-5)

    # Multivariate batch: (4, 10, 32)
    x_mul = torch.randn(4, 10, 32)
    model_mul = PatchCNN1D(in_channels=10, embed_dim=64, projection_dim=128)
    h_mul, z_mul = model_mul(x_mul)
    assert h_mul.shape == (4, 64)
    assert z_mul.shape == (4, 128)
    print("[PASS] PatchCNN1D forward test passed.")


def test_metric_learning_losses():
    z = F_normalize = torch.nn.functional.normalize(torch.randn(16, 64), dim=-1)
    indices = torch.arange(16)

    # Triplet loss
    triplet_loss = TemporalTripletLoss(margin=0.2, radius=2)
    l_triplet = triplet_loss(z, indices)
    assert l_triplet.ndim == 0
    assert not torch.isnan(l_triplet)

    # Contrastive loss
    contrastive_loss = TemporalContrastiveLoss(temperature=0.1, radius=2)
    l_contrast = contrastive_loss(z, indices)
    assert l_contrast.ndim == 0
    assert not torch.isnan(l_contrast)
    print("[PASS] Metric learning losses test passed.")


def test_synthetic_anomaly_detection():
    # Normal sine wave for training
    t = np.linspace(0, 20 * np.pi, 1000)
    train_x = np.sin(t)[:, np.newaxis]

    # Test sequence has normal sine wave + large anomalous burst at 500:600
    test_x = np.sin(t)[:, np.newaxis]
    test_labels = np.zeros(1000, dtype=np.int32)
    test_x[500:600] += 5.0
    test_labels[500:600] = 1

    detector = AnomlyticDetector(
        patch_size=32,
        embed_dim=32,
        projection_dim=64,
        num_iters=40,
        batch_size=64,
        lr=1e-3,
        device="cpu",
    )
    detector.fit(train_x)
    scores = detector.predict_score(test_x)
    metrics = detector.evaluate(scores, test_labels)

    assert metrics["AUC-ROC"] > 0.85, f"Expected AUC-ROC > 0.85, got {metrics['AUC-ROC']}"
    print(f"[PASS] Synthetic detection test passed (AUC-ROC: {metrics['AUC-ROC']:.4f}).")


def test_real_tsb_ad_sample():
    sample_path = Path("data/TSB-AD-U/013_NAB_id_13_Traffic_tr_623_1st_2084.csv")
    if not sample_path.exists():
        print("Real sample file not found, skipping real sample test.")
        return

    train_x, train_y, test_x, test_y = load_tsb_ad_file(sample_path)
    full_x = np.concatenate([train_x, test_x], axis=0)
    full_y = np.concatenate([train_y, test_y], axis=0)

    detector = AnomlyticDetector(
        patch_size=32,
        num_iters=30,
        batch_size=128,
        device="cpu",
    )
    detector.fit(train_x)
    scores = detector.predict_score(full_x)
    metrics = detector.evaluate(scores, full_y)

    assert 0.0 <= metrics["AUC-ROC"] <= 1.0
    print(f"[PASS] Real TSB-AD sample test passed: AUC-ROC = {metrics['AUC-ROC']:.4f}, AUC-PR = {metrics['AUC-PR']:.4f}")


if __name__ == "__main__":
    test_patch_extraction_and_distribution()
    test_patch_cnn_forward()
    test_metric_learning_losses()
    test_synthetic_anomaly_detection()
    test_real_tsb_ad_sample()
    print("\nAll Anomlytic tests passed successfully!")
