#!/usr/bin/env python
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.cluster import KMeans


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import DATA, RESULTS, sha256, write_json  # noqa: E402
from gaga import Autoencoder  # noqa: E402
from flow import ODEFunc, TimeSeriesDataset, infer  # noqa: E402


IN_DIR = DATA / "in_vivo"
OUT_DIR = RESULTS / "in_vivo" / "trajectories"

GROUPS = {
    "T1": {
        "csv": IN_DIR / "T1_flow_input.csv",
        "scaler": IN_DIR / "T1_flow_scaler.pkl",
        "checkpoint": IN_DIR / "T1_flow_model.pt",
    },
    "T2": {
        "csv": IN_DIR / "T2_flow_input.csv",
        "scaler": IN_DIR / "T2_flow_scaler.pkl",
        "checkpoint": IN_DIR / "T2_flow_model.pt",
    },
}


def build_dataset(frame: pd.DataFrame, scaler) -> TimeSeriesDataset:
    scaled = scaler.transform(frame[["d1", "d2", "d3"]].to_numpy())
    scaled_frame = pd.DataFrame(scaled, columns=["d1", "d2", "d3"])
    scaled_frame["samples"] = frame["samples"].to_numpy()
    series = []
    for sample in np.unique(scaled_frame["samples"]):
        points = scaled_frame.loc[
            scaled_frame["samples"] == sample, ["d1", "d2", "d3"]
        ].to_numpy()
        series.append((points, float(sample)))
    return TimeSeriesDataset(series)


def encode_cells() -> tuple[Autoencoder, dict]:
    adata = sc.read_h5ad(IN_DIR / "in_vivo.h5ad")
    with (IN_DIR / "gaga_pca_scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)
    model = Autoencoder(
        input_dim=adata.obsm["X_pca"].shape[1],
        latent_dim=3,
        hidden_dims=[256, 128, 64, 32],
    )
    model.load_state_dict(torch.load(IN_DIR / "gaga_model.pt", map_location="cpu"))
    model.eval()
    x_pca = scaler.transform(np.asarray(adata.obsm["X_pca"], dtype=np.float32)).astype(
        np.float32
    )
    with torch.no_grad():
        generated = model.encode(torch.from_numpy(x_pca)).numpy()
    np.save(OUT_DIR / "X_gaga.npy", generated)
    return model, {"shape": list(generated.shape)}


def integrate_group(name: str, spec: dict, indices: list[int]) -> tuple[np.ndarray, dict]:
    frame = pd.read_csv(spec["csv"]).sort_values("samples").reset_index(drop=True)
    with spec["scaler"].open("rb") as handle:
        scaler = pickle.load(handle)
    dataset = build_dataset(frame, scaler)
    initial = dataset.get_initial_condition()[torch.as_tensor(indices, dtype=torch.long)]
    model = ODEFunc(input_dim=3, hidden_dim=32).to("cpu")
    model.load_state_dict(torch.load(spec["checkpoint"], map_location="cpu"))
    model.eval()
    times = torch.linspace(min(dataset.times), max(dataset.times), 100, device="cpu")
    with torch.no_grad():
        scaled = infer(x0=initial, model=model, t_seq=times)
    scaled = scaled.permute(1, 0, 2).cpu().numpy()
    generated = scaler.inverse_transform(scaled.reshape(-1, 3)).reshape(scaled.shape).astype(
        np.float32
    )
    np.save(OUT_DIR / f"{name}_trajectories_latent.npy", generated)
    return generated, {"shape": list(generated.shape)}


def endpoint_labels(trajectory: np.ndarray) -> tuple[np.ndarray, dict]:
    labels = KMeans(n_clusters=3, random_state=42, n_init=1).fit_predict(
        trajectory[:, -1, :]
    )
    np.save(OUT_DIR / "T2_endpoint_labels.npy", labels)
    return labels, {"counts": {str(x): int((labels == x).sum()) for x in np.unique(labels)}}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    indices = json.loads((IN_DIR / "flow_initial_indices.json").read_text())
    _, gaga_report = encode_cells()
    latent = {}
    group_reports = {}
    for name, spec in GROUPS.items():
        latent[name], group_reports[name] = integrate_group(
            name, spec, indices["groups"][name]["indices"]
        )
    _, label_report = endpoint_labels(latent["T2"])
    report = {
        "gaga": gaga_report,
        "flow": group_reports,
        "T2_endpoint_labels": label_report,
        "inputs": {"in_vivo_h5ad_sha256": sha256(IN_DIR / "in_vivo.h5ad")},
    }
    write_json(OUT_DIR / "summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
