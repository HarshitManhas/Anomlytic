import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from .model import PatchCNN1D
from .patcher import extract_patches, distribute_scores_to_points
from .metric_learning import TemporalTripletLoss, TemporalContrastiveLoss
from .dataset import PatchDataset


class AnomlyticDetector:
    """
    End-to-End Anomaly Detector combining 1D CNN, Patch Representations, and Metric Learning.
    """

    def __init__(
        self,
        patch_size: int = 64,
        stride: int = 1,
        embed_dim: int = 64,
        projection_dim: int = 128,
        loss_type: str = "triplet",  # 'triplet' or 'contrastive'
        radius: int = 2,
        margin: float = 0.2,
        temperature: float = 0.1,
        num_iters: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        top_k: int = 3,
        memory_bank_ratio: float = 0.1,
        max_prototypes: int = 500,
        random_seed: int = 42,
        device: str | torch.device | None = None,
    ):
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim
        self.loss_type = loss_type
        self.radius = radius
        self.margin = margin
        self.temperature = temperature
        self.num_iters = num_iters
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.top_k = top_k
        self.memory_bank_ratio = memory_bank_ratio
        self.max_prototypes = max_prototypes
        self.random_seed = random_seed

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: PatchCNN1D | None = None
        self.memory_bank: torch.Tensor | None = None  # (K, embed_dim)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def _set_seed(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

    def fit(self, train_data: np.ndarray, verbose: bool = False) -> "AnomlyticDetector":
        """
        Trains the 1D CNN patch encoder on normal time series data via metric learning,
        and constructs the normal prototype memory bank.
        """
        self._set_seed()
        if train_data.ndim == 1:
            train_data = train_data[:, np.newaxis]

        # Learn global scaling parameters from normal training data
        self.mean = np.mean(train_data, axis=0, keepdims=True).astype(np.float32)
        self.std = np.std(train_data, axis=0, keepdims=True).astype(np.float32)
        self.std = np.where(self.std == 0.0, 1e-6, self.std)

        scaled_train = (train_data - self.mean) / self.std

        T, C = scaled_train.shape
        effective_patch_size = min(self.patch_size, T // 2 if T >= 4 else T)

        patches, indices = extract_patches(scaled_train, patch_size=effective_patch_size, stride=self.stride)
        N = len(patches)

        dataset = PatchDataset(patches, indices)
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, N),
            shuffle=True,
            drop_last=(N > self.batch_size),
        )

        # Initialize Model
        self.model = PatchCNN1D(
            in_channels=C,
            embed_dim=self.embed_dim,
            projection_dim=self.projection_dim,
            include_stats=True,
        ).to(self.device)

        # Loss Criterion
        if self.loss_type == "contrastive":
            criterion = TemporalContrastiveLoss(temperature=self.temperature, radius=self.radius)
        else:
            criterion = TemporalTripletLoss(margin=self.margin, radius=self.radius)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Cosine Annealed LR
        def lr_lambda(step: int) -> float:
            ratio = min(step / max(1, self.num_iters), 1.0)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * ratio))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        self.model.train()
        step = 0
        while step < self.num_iters:
            for batch_x, batch_idx in loader:
                if step >= self.num_iters:
                    break
                step += 1

                batch_x = batch_x.to(self.device)
                batch_idx = batch_idx.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                _, z = self.model(batch_x)
                loss = criterion(z, batch_idx)

                loss.backward()
                optimizer.step()
                scheduler.step()

                if verbose and step % max(1, self.num_iters // 5) == 0:
                    print(f"    [Step {step}/{self.num_iters}] Loss: {loss.item():.4f}")

        # Build Memory Bank of normal representations
        self._build_memory_bank(dataset)
        return self

    @torch.no_grad()
    def _build_memory_bank(self, dataset: PatchDataset):
        """Constructs a compact prototype memory bank from normal patch embeddings."""
        self.model.eval()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_feats = []
        for batch_x, _ in loader:
            batch_x = batch_x.to(self.device)
            feats = self.model.encode(batch_x)
            all_feats.append(feats.cpu())

        feats_tensor = torch.cat(all_feats, dim=0)  # (N, D)
        feats_norm = F.normalize(feats_tensor, p=2, dim=-1)

        N = feats_norm.size(0)
        target_k = int(round(self.memory_bank_ratio * N))
        num_clusters = max(5, min(target_k, self.max_prototypes, N - 1 if N > 1 else 1))

        if num_clusters >= N:
            self.memory_bank = feats_norm.to(self.device)
            return

        mbk = MiniBatchKMeans(
            n_clusters=num_clusters,
            random_state=self.random_seed,
            batch_size=min(1024, N),
            n_init=1,
            max_iter=50,
        )
        mbk.fit(feats_norm.numpy())
        centers = torch.from_numpy(mbk.cluster_centers_).float()
        dist_matrix = torch.cdist(feats_norm, centers)
        medoid_indices = torch.argmin(dist_matrix, dim=0)

        prototypes = feats_norm[medoid_indices]
        self.memory_bank = prototypes.to(self.device)

    @torch.no_grad()
    def predict_score(self, data: np.ndarray) -> np.ndarray:
        """
        Computes point-wise anomaly scores for an input time series.
        Returns:
            scores: 1D numpy array of shape (T,) with scores in [0, 1] range.
        """
        if self.model is None or self.memory_bank is None:
            raise RuntimeError("Detector must be fitted before predict_score.")

        self.model.eval()
        if data.ndim == 1:
            data = data[:, np.newaxis]

        # Apply same scaling learned during fit
        scaled_data = (data - self.mean) / self.std

        T, _ = scaled_data.shape
        effective_patch_size = min(self.patch_size, T // 2 if T >= 4 else T)

        patches, _ = extract_patches(scaled_data, patch_size=effective_patch_size, stride=self.stride)
        dataset = PatchDataset(patches, np.arange(len(patches)))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        patch_scores = []
        k = min(self.top_k, self.memory_bank.size(0))

        for batch_x, _ in loader:
            batch_x = batch_x.to(self.device)
            feats = self.model.encode(batch_x)
            feats = F.normalize(feats, p=2, dim=-1)  # (B, D)

            sims = torch.matmul(feats, self.memory_bank.T)  # (B, M)
            top_sims, _ = torch.topk(sims, k=k, dim=1, largest=True)
            dists = 1.0 - top_sims
            score_batch = dists.mean(dim=1)
            patch_scores.extend(score_batch.cpu().tolist())

        point_scores = distribute_scores_to_points(
            np.array(patch_scores),
            patch_size=effective_patch_size,
            num_points=T,
            stride=self.stride,
        )
        return point_scores

    @staticmethod
    def evaluate(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """
        Evaluates detection performance against ground truth labels.
        Returns AUC-ROC, AUC-PR, Best-F1, Precision, and Recall.
        """
        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
        labels = np.asarray(labels, dtype=np.int32)

        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return {
                "AUC-ROC": 0.5,
                "AUC-PR": float(labels.mean()),
                "Best-F1": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
            }

        auc_roc = float(roc_auc_score(labels, scores))
        auc_pr = float(average_precision_score(labels, scores))

        precisions, recalls, thresholds = precision_recall_curve(labels, scores)
        f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1s)

        return {
            "AUC-ROC": round(auc_roc, 4),
            "AUC-PR": round(auc_pr, 4),
            "Best-F1": round(float(f1s[best_idx]), 4),
            "Precision": round(float(precisions[best_idx]), 4),
            "Recall": round(float(recalls[best_idx]), 4),
        }
