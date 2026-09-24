"""Neural policy for constrained adaptive per-example clipping."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AdaptiveClippingPredictor(nn.Module):
    """Encode projected-gradient history and projected model weights."""

    def __init__(
        self,
        gradient_dim: int = 4_275,
        norm_dim: int = 6,
        weight_dim: int = 512,
        token_dim: int = 128,
        history_dim: int = 128,
        weight_hidden_dim: int = 128,
        weight_embedding_dim: int = 64,
        policy_hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gradient_dim = gradient_dim
        self.norm_dim = norm_dim
        self.weight_dim = weight_dim

        self.token_encoder = nn.Sequential(
            nn.Linear(gradient_dim + norm_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.history_encoder = nn.GRU(
            input_size=token_dim,
            hidden_size=history_dim,
            batch_first=True,
        )
        self.weight_encoder = nn.Sequential(
            nn.Linear(weight_dim, weight_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(weight_hidden_dim, weight_embedding_dim),
            nn.GELU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(history_dim + weight_embedding_dim + 1, policy_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(policy_hidden_dim, 1),
        )

    def forward(
        self,
        history_sketch: Tensor,
        history_norms: Tensor,
        weight_projection: Tensor,
        normalized_step: Tensor,
        sketch_mean: Tensor,
        sketch_std: Tensor,
        norm_mean: Tensor,
        norm_std: Tensor,
        weight_mean: Tensor,
        weight_std: Tensor,
    ) -> Tensor:
        """Return the logit for choosing the low clipping action."""
        sketches = (history_sketch.float() - sketch_mean) / sketch_std
        norms = (torch.log1p(history_norms.float()) - norm_mean) / norm_std
        tokens = self.token_encoder(torch.cat((sketches, norms), dim=-1))
        _, hidden = self.history_encoder(tokens)
        history_embedding = hidden[-1]

        weights = (weight_projection.float() - weight_mean) / weight_std
        weight_embedding = self.weight_encoder(weights)
        if normalized_step.ndim == 1:
            normalized_step = normalized_step.unsqueeze(-1)
        fused = torch.cat(
            (history_embedding, weight_embedding, normalized_step.float()), dim=-1
        )
        return self.policy_head(fused).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
