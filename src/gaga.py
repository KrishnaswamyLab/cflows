"""GAGA model."""
from __future__ import annotations

import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list[int]):
        super().__init__()
        encoder = []
        previous = input_dim
        for width in hidden_dims:
            encoder.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        encoder.append(nn.Linear(previous, latent_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        previous = latent_dim
        for width in reversed(hidden_dims):
            decoder.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        decoder.append(nn.Linear(previous, input_dim))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        return self.encoder(value)

    def decode(self, value: torch.Tensor) -> torch.Tensor:
        return self.decoder(value)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(value))
