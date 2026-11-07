import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalTripletLoss(nn.Module):
    """
    Temporal Triplet Loss with hard negative mining.
    - Positive: random patch within temporal radius r (|idx_a - idx_p| <= r, idx_a != idx_p).
    - Hard Negative: out-of-radius patch in the batch with smallest distance to anchor.
    """

    def __init__(self, margin: float = 0.2, radius: int = 2):
        super().__init__()
        self.margin = margin
        self.radius = radius

    def forward(
        self,
        z: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z: L2-normalized projections (B, D)
            indices: 1D tensor of patch time indices (B,)
        """
        B = z.size(0)
        if B < 2:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        indices = indices.view(-1)
        # Pairwise temporal distance: |idx_i - idx_j|
        time_diff = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))  # (B, B)

        # Pairwise cosine distance: 1 - z_i . z_j (since z is L2 normalized)
        sim = torch.matmul(z, z.T)  # (B, B)
        dist = 1.0 - sim

        # Mask for valid positives: within radius and not self
        pos_mask = (time_diff > 0) & (time_diff <= self.radius)
        # Mask for negatives: outside radius
        neg_mask = time_diff > self.radius

        # If an anchor has no positive in batch, fallback to diagonal self-dist
        has_pos = pos_mask.any(dim=1)
        has_neg = neg_mask.any(dim=1)
        valid_anchors = has_pos & has_neg

        if not valid_anchors.any():
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        # Positive distances: take mean or min positive distance
        pos_dists = torch.where(pos_mask, dist, torch.tensor(float("inf"), device=z.device))
        pos_dist, _ = torch.min(pos_dists, dim=1)

        # Hardest negative distances: smallest distance among negatives
        neg_dists = torch.where(neg_mask, dist, torch.tensor(float("inf"), device=z.device))
        hard_neg_dist, _ = torch.min(neg_dists, dim=1)

        # Triplet loss on valid anchors
        loss = F.relu(pos_dist[valid_anchors] - hard_neg_dist[valid_anchors] + self.margin).mean()
        return loss


class TemporalContrastiveLoss(nn.Module):
    """
    NT-Xent style Temporal Contrastive Loss.
    Pulls temporal neighbors together and repels distant patches with temperature scaling.
    """

    def __init__(self, temperature: float = 0.1, radius: int = 2):
        super().__init__()
        self.temperature = temperature
        self.radius = radius

    def forward(
        self,
        z: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        B = z.size(0)
        if B < 2:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        indices = indices.view(-1)
        time_diff = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))  # (B, B)

        # Similarities scaled by temperature
        logits = torch.matmul(z, z.T) / self.temperature  # (B, B)

        # Remove diagonal (self-similarity)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()  # numerical stability

        pos_mask = ((time_diff > 0) & (time_diff <= self.radius)).float()
        neg_mask = (time_diff > self.radius).float()

        # Anchors with at least one positive and one negative
        valid = (pos_mask.sum(dim=1) > 0) & (neg_mask.sum(dim=1) > 0)
        if not valid.any():
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        # Log-sum-exp over negatives + positives
        exp_logits = torch.exp(logits) * (pos_mask + neg_mask)
        denominator = exp_logits.sum(dim=1, keepdim=True)

        # Log probability of positives
        log_prob = logits - torch.log(denominator + 1e-8)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)

        loss = -mean_log_prob_pos[valid].mean()
        return loss
