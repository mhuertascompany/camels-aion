"""Model definitions for CNN/ViT baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models


class CNNBaseline(nn.Module):
    """Simple convolutional network for CAMELS regression."""

    def __init__(self, in_channels: int, num_outputs: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, num_outputs)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats = self.features(x).flatten(1)
        preds = self.head(feats)
        if return_features:
            return preds, feats
        return preds


class ViTBaseline(nn.Module):
    """Wrapper around torchvision ViT for four-channel inputs."""

    def __init__(self, in_channels: int, num_outputs: int) -> None:
        super().__init__()
        self.model = models.vit_b_16(weights=None)
        conv = self.model.conv_proj
        self.model.conv_proj = nn.Conv2d(
            in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=conv.bias is not None,
        )
        nn.init.kaiming_normal_(self.model.conv_proj.weight, mode="fan_out", nonlinearity="relu")
        if conv.bias is not None:
            nn.init.zeros_(self.model.conv_proj.bias)

        embed_dim = self.model.heads.head.in_features
        self.model.heads = nn.Identity()
        self.head = nn.Linear(embed_dim, num_outputs)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats = self.model(x)
        preds = self.head(feats)
        if return_features:
            return preds, feats
        return preds


def build_model(model_type: str, in_channels: int, num_outputs: int) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "cnn":
        return CNNBaseline(in_channels, num_outputs)
    if model_type == "vit":
        return ViTBaseline(in_channels, num_outputs)
    raise ValueError(f"Unknown model type '{model_type}'")
