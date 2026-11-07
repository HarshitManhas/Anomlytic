import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchCNN1D(nn.Module):
    """
    1D CNN Backbone + Projection Head for temporal patch metric learning.
    Captures both local temporal patterns (via 1D Conv) and scale/level dynamics (via patch statistics).
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 64,
        projection_dim: int = 128,
        channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        include_stats: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim
        self.include_stats = include_stats
        self.eps = eps

        channels = channels or [64, 128, 128, 64]
        kernel_sizes = kernel_sizes or [7, 5, 3, 3]

        layers = []
        prev_c = in_channels
        for c, k in zip(channels, kernel_sizes):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(prev_c, c, kernel_size=k, stride=1, padding=k // 2, bias=False),
                    nn.BatchNorm1d(c),
                    nn.GELU(),
                )
            )
            prev_c = c

        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Map combined features (conv + stats) to embedding space
        fc_in = channels[-1] + (2 * in_channels if include_stats else 0)
        self.fc_embed = nn.Linear(fc_in, embed_dim)

        # 2-layer MLP projection head for metric learning
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts latent patch representation h (batch_size, embed_dim)."""
        # x shape: (B, C, L)
        m = x.mean(dim=-1)  # (B, C)
        s = x.std(dim=-1, unbiased=False)  # (B, C)

        # Scale-normalized patch for convolutional feature extraction
        x_norm = (x - m.unsqueeze(-1)) / (s.unsqueeze(-1) + self.eps)
        conv_feat = self.pool(self.backbone(x_norm)).flatten(start_dim=1)  # (B, channels[-1])

        if self.include_stats:
            combined = torch.cat([conv_feat, m, s], dim=1)
        else:
            combined = conv_feat

        return self.fc_embed(combined)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """Projects latent representation into L2-normalized metric learning space."""
        z = self.projection_head(h)
        return F.normalize(z, p=2, dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        z = self.project(h)
        return h, z
