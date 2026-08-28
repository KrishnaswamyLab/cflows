"""Flow and growth-rate models."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, momentum_beta: float = 0.0):
        super().__init__()
        self.momentum_beta = momentum_beta
        self.previous_v = None
        self.model = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        for layer in self.model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def reset_momentum(self) -> None:
        self.previous_v = None

    def forward(self, time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        expanded_time = time.expand(value.size(0), 1)
        velocity = self.model(torch.cat((expanded_time, value), dim=-1))
        if self.momentum_beta > 0.0:
            if self.previous_v is None or self.previous_v.shape[0] != value.shape[0]:
                self.previous_v = torch.zeros_like(velocity)
            velocity = self.momentum_beta * self.previous_v + (1 - self.momentum_beta) * velocity
            self.previous_v = velocity.detach()
        return velocity


class GrowthRateModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, use_time: bool = True):
        super().__init__()
        self.use_time = use_time
        actual_input_dim = input_dim + 1 if use_time else input_dim
        self.net = nn.Sequential(
            nn.Linear(actual_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.softplus = nn.Softplus()
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 0.5413)

    def forward(self, value: torch.Tensor, time: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_time:
            if time is None:
                raise ValueError("time is required when use_time=True")
            if time.dim() == 0:
                expanded_time = time.expand(value.size(0), 1)
            elif time.dim() == 1 and time.size(0) == 1:
                expanded_time = time.expand(value.size(0), 1)
            else:
                expanded_time = time.view(-1, 1)
            value = torch.cat((value, expanded_time), dim=-1)
        return self.softplus(self.net(value)).squeeze(-1) + 1e-4


def infer(x0: torch.Tensor, model: ODEFunc, t_seq: torch.Tensor) -> torch.Tensor:
    model.reset_momentum()
    return odeint(model, x0, t_seq, method="dopri5", rtol=1e-7, atol=1e-9)


class TimeSeriesDataset(Dataset):
    def __init__(self, time_series_data: list[tuple[np.ndarray, float]]):
        self.time_series_data = time_series_data
        self.times = [time for _, time in time_series_data]

    def __len__(self) -> int:
        return len(self.time_series_data) - 1

    def __getitem__(self, index: int) -> dict:
        start, t_start = self.time_series_data[index]
        end, t_end = self.time_series_data[index + 1]
        return {
            "X_start": torch.tensor(start, dtype=torch.float32),
            "X_end": torch.tensor(end, dtype=torch.float32),
            "t_start": t_start,
            "t_end": t_end,
            "interval_idx": index,
        }

    def get_initial_condition(self, start_idx: int = 0) -> torch.Tensor:
        value, _ = self.time_series_data[start_idx]
        return torch.tensor(value, dtype=torch.float32)
