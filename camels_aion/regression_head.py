"""Pooling and regression head utilities for AION embeddings."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class TokenPooler(nn.Module):
    def __init__(
        self,
        pool_type: str,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        pool_type = pool_type.lower()
        if pool_type not in {"mean", "meanmax", "attention"}:
            raise ValueError(f"Unknown pooling type '{pool_type}'")
        self.pool_type = pool_type
        self.num_heads = num_heads
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if pool_type == "attention":
            self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.query, std=1e-2)
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            self.output_dim = embed_dim
        elif pool_type == "meanmax":
            self.output_dim = embed_dim * 2
        else:
            self.output_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            pooled = x
        else:
            if self.pool_type == "mean":
                pooled = x.mean(dim=1)
            elif self.pool_type == "meanmax":
                mean = x.mean(dim=1)
                maxv = x.max(dim=1).values
                pooled = torch.cat([mean, maxv], dim=1)
            elif self.pool_type == "attention":
                batch_size = x.shape[0]
                query = self.query.expand(batch_size, -1, -1)
                attn_out, _ = self.attn(query, x, x)
                pooled = attn_out.squeeze(1)
            else:
                raise RuntimeError("Unsupported pooling type")
        return self.dropout_layer(pooled)


class RegressionModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_outputs: int,
        hidden_dims: Sequence[int] | None = None,
        dropout: float = 0.2,
        pool: TokenPooler | None = None,
        pool_type: str = "mean",
        pool_heads: int = 4,
        pool_dropout: float = 0.1,
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = tuple(hidden_dims or [])

        self.pool = pool if pool is not None else TokenPooler(pool_type, input_dim, pool_heads, pool_dropout)
        feature_dim = self.pool.output_dim

        self.register_buffer("feature_mean", torch.zeros(1, feature_dim))
        self.register_buffer("feature_std", torch.ones(1, feature_dim))
        self.set_normalization_stats(feature_mean, feature_std)

        layers: list[nn.Module] = []
        prev_dim = feature_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, prev_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            #prev_dim = hidden_dim
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
        if x.dim() == 3:
            pooled = self.pool(x)
        else:
            pooled = x
        normed = (pooled - self.feature_mean) / self.feature_std
        #normed = pooled
        h = self.backbone(normed) if len(self.backbone) > 0 else normed
        preds = self.head(h)
        if return_features:
            return preds, h
        return preds
