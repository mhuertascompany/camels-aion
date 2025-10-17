"""Regression head with feature standardization and configurable MLP."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class RegressionModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_outputs: int,
        hidden_dims: Sequence[int] | None = None,
        dropout: float = 0.5,
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = tuple(hidden_dims or [])

        self.register_buffer("feature_mean", torch.zeros(1, input_dim))
        self.register_buffer("feature_std", torch.ones(1, input_dim))
        self.set_normalization_stats(feature_mean, feature_std)

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, num_outputs)

    def set_normalization_stats(
        self,
        feature_mean: torch.Tensor | None,
        feature_std: torch.Tensor | None,
    ) -> None:
        if feature_mean is not None:
            mean = feature_mean.detach().to(self.feature_mean.dtype)
            if mean.ndim == 1:
                mean = mean.unsqueeze(0)
            self.feature_mean.copy_(mean)
        if feature_std is not None:
            std = feature_std.detach().to(self.feature_std.dtype)
            std = torch.clamp(std, min=1e-6)
            if std.ndim == 1:
                std = std.unsqueeze(0)
            self.feature_std.copy_(std)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = (x - self.feature_mean) / self.feature_std
        h = self.backbone(x) if len(self.backbone) > 0 else x
        preds = self.head(h)
        if return_features:
            return preds, h
        return preds
