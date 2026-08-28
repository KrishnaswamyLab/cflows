#!/usr/bin/env python
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import DATA, RESULTS, write_json  # noqa: E402
from flow import GrowthRateModel  # noqa: E402


IN_DIR = DATA / "in_vivo"
TRAJ_DIR = RESULTS / "in_vivo" / "trajectories"
OUT_DIR = RESULTS / "in_vivo" / "growth_filter"


GROUPS = {
    "T1": {
        "csv": IN_DIR / "T1_flow_input.csv",
        "scaler": IN_DIR / "T1_flow_scaler.pkl",
        "checkpoint": IN_DIR / "T1_growth_model.pt",
        "trajectory": TRAJ_DIR / "T1_trajectories_latent.npy",
    },
    "T2": {
        "csv": IN_DIR / "T2_flow_input.csv",
        "scaler": IN_DIR / "T2_flow_scaler.pkl",
        "checkpoint": IN_DIR / "T2_growth_model.pt",
        "trajectory": TRAJ_DIR / "T2_trajectories_latent.npy",
    },
}


def gmm_threshold(values: np.ndarray) -> dict:
    x = values.reshape(-1, 1)
    two = GaussianMixture(n_components=2, random_state=42, n_init=10).fit(x)
    order = np.argsort(two.means_.reshape(-1))
    means = two.means_.reshape(-1)[order]
    variances = two.covariances_.reshape(-1)[order]
    weights = two.weights_.reshape(-1)[order]
    grid = np.linspace(values.min(), values.max(), 2000)
    components = np.zeros((2, len(grid)))
    for i in range(2):
        sd = np.sqrt(variances[i])
        components[i] = weights[i] * np.exp(-0.5 * ((grid - means[i]) / sd) ** 2) / (
            sd * np.sqrt(2 * np.pi)
        )
    difference = components[0] - components[1]
    crossings = np.where(np.sign(difference[:-1]) != np.sign(difference[1:]))[0]
    between = crossings[(grid[crossings] >= means[0]) & (grid[crossings] <= means[1])]
    if not len(between):
        raise RuntimeError("no GMM intersection between component means")
    i = int(between[np.argmin(np.abs(grid[between] - means.mean()))])
    x0, x1 = grid[i], grid[i + 1]
    y0, y1 = difference[i], difference[i + 1]
    threshold = float(x0 - y0 * (x1 - x0) / (y1 - y0))
    return {
        "threshold": threshold,
        "component_means": means.tolist(),
        "component_sds": np.sqrt(variances).tolist(),
        "component_weights": weights.tolist(),
    }


def process(name: str, spec: dict) -> dict:
    frame = pd.read_csv(spec["csv"]).sort_values("samples").reset_index(drop=True)
    with spec["scaler"].open("rb") as handle:
        scaler = pickle.load(handle)
    trajectory = np.load(spec["trajectory"])
    model = GrowthRateModel(input_dim=3, hidden_dim=32, use_time=True)
    model.load_state_dict(torch.load(spec["checkpoint"], map_location="cpu"))
    model.eval()
    x = torch.tensor(scaler.transform(trajectory[:, 0, :]), dtype=torch.float32)
    t = torch.full((x.shape[0], 1), float(frame["samples"].min()), dtype=torch.float32)
    with torch.no_grad():
        growth = model(x, t).cpu().numpy().reshape(-1)
    gmm = gmm_threshold(growth)
    mask = growth >= gmm["threshold"]
    np.save(OUT_DIR / f"{name}_to_grow_mask.npy", mask)
    pd.DataFrame(
        {
            "trajectory_index": np.arange(len(growth)),
            "growth_pred_start": growth,
            "gmm_threshold": gmm["threshold"],
            "start_group": np.where(mask, "to_grow", "to_die"),
        }
    ).to_csv(OUT_DIR / f"{name}_growth_labels.csv", index=False)
    return {
        "n_trajectories": int(len(mask)),
        "n_to_grow": int(mask.sum()),
        "min_margin_to_threshold": float(np.min(np.abs(growth - gmm["threshold"]))),
        "gmm": gmm,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {name: process(name, spec) for name, spec in GROUPS.items()}
    write_json(OUT_DIR / "summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
