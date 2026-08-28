"""Decoder model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer_widths: list[int],
        dropout: float,
        batchnorm: bool,
        **_: object,
    ):
        super().__init__()
        layers = []
        previous = input_dim
        for width in layer_widths:
            layers.append(nn.Linear(previous, width))
            if batchnorm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def predict(
    value: np.ndarray, checkpoint_path: str | Path, scaler_path: str | Path
) -> np.ndarray:
    scaler = np.load(scaler_path)
    scaled = (value - scaler["mean"]) / scaler["scale"]
    tensor = torch.tensor(scaled, dtype=torch.float32)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = Decoder(**checkpoint["hyper_parameters"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        return model(tensor).numpy()
