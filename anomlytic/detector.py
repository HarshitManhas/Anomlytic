import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import copy
import math
import time
from pathlib import Path
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
    Supports checkpointing, train/val loss recording, and inference latency/VRAM profiling.
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
        self.history: dict[str, list[float]] = {"step": [], "train_loss": [], "val_loss": []}

    def _set_seed(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

    def fit(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray | None = None,
        checkpoint_path: str | Path | None = None,
        verbose: bool = False,
    ) -> "AnomlyticDetector":
        """
        Trains the 1D CNN patch encoder on normal time series data via metric learning,
        optionally computes validation loss, and saves model checkpoints.
        """
        self._set_seed()
        self.history = {"step": [], "train_loss": [], "val_loss": []}

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

        # Prepare Validation Loader if val_data provided
        val_loader = None
        if val_data is not None:
            if val_data.ndim == 1:
                val_data = val_data[:, np.newaxis]
            scaled_val = (val_data - self.mean) / self.std
            val_patches, val_indices = extract_patches(scaled_val, patch_size=effective_patch_size, stride=self.stride)
            val_dataset = PatchDataset(val_patches, val_indices)
            val_loader = DataLoader(val_dataset, batch_size=min(self.batch_size, len(val_patches)), shuffle=False)

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

        best_loss = float("inf")
        best_state_dict = None
        step = 0

        while step < self.num_iters:
            for batch_x, batch_idx in loader:
                if step >= self.num_iters:
                    break
                step += 1

                self.model.train()
                batch_x = batch_x.to(self.device)
                batch_idx = batch_idx.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                _, z = self.model(batch_x)
                loss = criterion(z, batch_idx)

                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss_val = float(loss.item())

                # Compute validation loss
                val_loss_val = None
                if val_loader is not None:
                    self.model.eval()
                    with torch.no_grad():
                        val_losses = []
                        for vx, vi in val_loader:
                            vx = vx.to(self.device)
                            vi = vi.to(self.device)
                            _, vz = self.model(vx)
                            val_losses.append(criterion(vz, vi).item())
                        val_loss_val = float(np.mean(val_losses)) if val_losses else train_loss_val

                self.history["step"].append(step)
                self.history["train_loss"].append(train_loss_val)
                if val_loss_val is not None:
                    self.history["val_loss"].append(val_loss_val)

                monitored_loss = val_loss_val if val_loss_val is not None else train_loss_val
                if monitored_loss < best_loss:
                    best_loss = monitored_loss
                    best_state_dict = copy.deepcopy(self.model.state_dict())

                if verbose and step % max(1, self.num_iters // 5) == 0:
                    v_str = f" | Val Loss: {val_loss_val:.4f}" if val_loss_val is not None else ""
                    print(f"    [Step {step}/{self.num_iters}] Train Loss: {train_loss_val:.4f}{v_str}")

        # Restore weights that achieved best loss
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        # Build Memory Bank of normal representations
        self._build_memory_bank(dataset)

        # Save checkpoint if requested
        if checkpoint_path:
            self.save_checkpoint(checkpoint_path)

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

    def profile_inference(self, data: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """
        Profiles inference execution: measures point-wise anomaly score generation,
        wall-clock latency (ms), throughput (microseconds/point), and peak GPU VRAM (MB).
        """
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

        t_start = time.perf_counter()
        scores = self.predict_score(data)

        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        else:
            peak_vram_mb = 0.0

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        num_points = len(data)
        us_per_point = (latency_ms * 1000.0) / max(1, num_points)

        profile = {
            "Total_Latency_ms": round(latency_ms, 2),
            "Latency_us_per_point": round(us_per_point, 2),
            "Peak_GPU_Memory_MB": round(peak_vram_mb, 2),
            "Num_Points": num_points,
        }
        return scores, profile

    def save_checkpoint(self, path: str | Path):
        """Saves complete model checkpoint, config, scalers, and memory bank."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state": self.model.state_dict() if self.model else None,
            "memory_bank": self.memory_bank,
            "mean": self.mean,
            "std": self.std,
            "history": self.history,
            "config": {
                "patch_size": self.patch_size,
                "stride": self.stride,
                "embed_dim": self.embed_dim,
                "projection_dim": self.projection_dim,
                "loss_type": self.loss_type,
                "radius": self.radius,
                "margin": self.margin,
                "temperature": self.temperature,
                "num_iters": self.num_iters,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "batch_size": self.batch_size,
                "top_k": self.top_k,
                "memory_bank_ratio": self.memory_bank_ratio,
                "max_prototypes": self.max_prototypes,
                "random_seed": self.random_seed,
            },
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path):
        """Loads model weights, scalers, and memory bank from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]
        if checkpoint["memory_bank"] is not None:
            self.memory_bank = checkpoint["memory_bank"].to(self.device)
        else:
            self.memory_bank = None
        self.history = checkpoint.get("history", {})

        cfg = checkpoint.get("config", {})
        for k, v in cfg.items():
            if hasattr(self, k):
                setattr(self, k, v)

        if checkpoint["model_state"] and self.model is None:
            # Infer in_channels from mean shape
            in_c = self.mean.shape[1] if self.mean is not None else 1
            self.model = PatchCNN1D(
                in_channels=in_c,
                embed_dim=self.embed_dim,
                projection_dim=self.projection_dim,
                include_stats=True,
            ).to(self.device)

        if self.model and checkpoint["model_state"]:
            self.model.load_state_dict(checkpoint["model_state"])

    @staticmethod
    def evaluate(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """
        Evaluates detection performance against ground truth labels.
        Returns Precision, Recall, F1-score, PR-AUC, ROC-AUC, and optimal Threshold.
        """
        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
        labels = np.asarray(labels, dtype=np.int32)

        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return {
                "Precision": 0.0,
                "Recall": 0.0,
                "F1-score": 0.0,
                "PR-AUC": float(labels.mean()),
                "ROC-AUC": 0.5,
                "Threshold": 0.5,
            }

        auc_roc = float(roc_auc_score(labels, scores))
        auc_pr = float(average_precision_score(labels, scores))

        # Precision-Recall curve threshold sweep for best F1
        precisions, recalls, thresholds = precision_recall_curve(labels, scores)
        f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = int(np.argmax(f1s))

        best_precision = float(precisions[best_idx])
        best_recall = float(recalls[best_idx])
        best_f1 = float(f1s[best_idx])
        best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else float(thresholds[-1])

        return {
            "Precision": round(best_precision, 4),
            "Recall": round(best_recall, 4),
            "F1-score": round(best_f1, 4),
            "PR-AUC": round(auc_pr, 4),
            "ROC-AUC": round(auc_roc, 4),
            "Threshold": round(best_thresh, 4),
        }
