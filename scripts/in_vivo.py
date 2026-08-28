#!/usr/bin/env python
"""In vivo: produce Figure 7b-d and 7f."""
from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import DATA, RESULTS, effect_size_table, n_jobs, qc_gene_mask, save_granger, write_json  # noqa: E402
from gaga import Autoencoder  # noqa: E402
from granger import do_granger  # noqa: E402


IN_DIR = DATA / "in_vivo"
TRAJ_DIR = RESULTS / "in_vivo" / "trajectories"
FILTER_DIR = RESULTS / "in_vivo" / "growth_filter"
OUT_DIR = RESULTS / "in_vivo"
GRANGER_DIR = OUT_DIR / "granger"
FIGURE_DIR = OUT_DIR / "figures"

T1_COLOR = "#E6A024"
T2_COLOR = "#7B2D6E"
FRAME_COLOR = "#C8C8C8"
ELEV, AZIM = 30, 330
T2_LABEL = 2
N_PLOT_OUTLIERS = 6
CLUSTER_COLORS = {
    "other": "#B0BEC5",
    "start": "#5C6BC0",
    "T1_end": T1_COLOR,
    "T2_end": T2_COLOR,
}
CLUSTER_ALPHA = {"other": 0.08, "start": 0.65, "T1_end": 0.90, "T2_end": 0.90}
CLUSTER_SIZE = {"other": 1.5, "start": 6, "T1_end": 14, "T2_end": 14}
LEGEND_KW = {
    "fontsize": 8,
    "framealpha": 0.6,
    "loc": "upper left",
    "bbox_to_anchor": (1.02, 1.0),
    "handlelength": 1.4,
    "handletextpad": 0.5,
    "borderpad": 0.5,
}


def load_decoder(adata):
    with (IN_DIR / "gaga_pca_scaler.pkl").open("rb") as handle:
        pca_scaler = pickle.load(handle)
    model = Autoencoder(
        input_dim=adata.obsm["X_pca"].shape[1], latent_dim=3, hidden_dims=[256, 128, 64, 32]
    )
    model.load_state_dict(torch.load(IN_DIR / "gaga_model.pt", map_location="cpu"))
    model.eval()
    means = adata.X.mean(axis=0)
    means = means.A1 if hasattr(means, "A1") else np.asarray(means).ravel()
    return model, pca_scaler, np.asarray(adata.varm["PCs"]), means


def decode_selected(latent, indices, model, pca_scaler, pcs, means) -> np.ndarray:
    shape = latent.shape
    with torch.no_grad():
        pca_scaled = model.decode(torch.tensor(latent.reshape(-1, 3), dtype=torch.float32)).numpy()
    pca = pca_scaler.inverse_transform(pca_scaled)
    gene = pca @ pcs[indices].T + means[indices]
    return gene.reshape(shape[0], shape[1], len(indices))


def chronological_labels(raw: np.ndarray) -> np.ndarray:
    seen = []
    for label in raw:
        if int(label) not in seen:
            seen.append(int(label))
    remap = {old: new for new, old in enumerate(seen)}
    return np.array([remap[int(label)] for label in raw], dtype=int)


def cluster_trends(trajectory: np.ndarray, genes: np.ndarray, k: int):
    mean = trajectory.mean(axis=0)
    varying = mean.std(axis=0) != 0
    mean, genes = mean[:, varying], genes[varying]
    normalized = (mean - mean.min(axis=0)) / (mean.max(axis=0) - mean.min(axis=0))
    trends = pd.DataFrame(
        normalized.T, index=genes, columns=[f"t{i}" for i in range(normalized.shape[0])]
    )
    peak = trends.apply(lambda row: row.argsort()[-1:].mean(), axis=1)
    order = peak.argsort()
    trends = trends.iloc[order]
    sorted_peak = peak.iloc[order].to_numpy()
    raw = KMeans(n_clusters=k, random_state=42, n_init=1).fit_predict(
        sorted_peak.reshape(-1, 1)
    )
    cluster = chronological_labels(raw)
    table = pd.DataFrame(
        {
            "number": np.arange(len(trends)),
            "cluster": cluster,
            "peak_index": sorted_peak,
            "sort_order": np.arange(len(trends)),
        },
        index=trends.index,
    )
    return table, trends


def run_granger_group(name, trajectory, genes, hvg, tfs):
    valid = trajectory.mean(axis=0).var(axis=0) != 0
    trajectory, genes = trajectory[:, :, valid], genes[valid]
    frame = pd.DataFrame(trajectory.mean(axis=0), columns=genes)
    present = set(genes)
    tf_kept = [gene for gene in tfs if gene in present]
    hvg_kept = [gene for gene in hvg if gene in present]
    pvals, coefs = do_granger(frame.T, in_genes=tf_kept, out_genes=hvg_kept, n_jobs=n_jobs())
    signed = save_granger(GRANGER_DIR / f"{name}_granger", pvals, coefs)
    return pvals, coefs, signed


def finalise_3d(axis):
    axis.view_init(elev=ELEV, azim=AZIM)
    transparent = (1.0, 1.0, 1.0, 0.0)
    for item in (axis.xaxis, axis.yaxis, axis.zaxis):
        item.set_ticks([])
        item.label.set_visible(False)
        item.pane.set_facecolor(transparent)
        item.pane.set_edgecolor(transparent)
        item.line.set_color(transparent)
    axis.grid(False)
    xlo, xhi = axis.get_xlim3d()
    ylo, yhi = axis.get_ylim3d()
    zlo, zhi = axis.get_zlim3d()
    corners = np.array([
        (xlo, ylo, zlo), (xhi, ylo, zlo), (xhi, yhi, zlo), (xlo, yhi, zlo),
        (xlo, ylo, zhi), (xhi, ylo, zhi), (xhi, yhi, zhi), (xlo, yhi, zhi),
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    view = np.array([
        np.cos(np.radians(ELEV)) * np.cos(np.radians(AZIM)),
        np.cos(np.radians(ELEV)) * np.sin(np.radians(AZIM)),
        np.sin(np.radians(ELEV)),
    ])
    nearest = int(np.argmax((corners - corners.mean(axis=0)) @ view))
    for i, j in edges:
        if i != nearest and j != nearest:
            a, b = corners[i], corners[j]
            axis.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                      color=FRAME_COLOR, lw=0.7, zorder=0)


def evenly_spaced(indices, n=25):
    if len(indices) <= n:
        return indices
    return indices[np.linspace(0, len(indices) - 1, n).round().astype(int)]


def plot_figure7bcd(adata, x_gaga, t1, t2, t1_mask, t2_mask, labels):
    sample_map = {"TP2_PT": "Day 42", "TP3_V": "Day 49", "TP5_V": "Day 56"}
    sample_colors = {"Day 42": "#E6A024", "Day 49": "#CC79A7", "Day 56": "#0273B3"}
    sample = np.array([sample_map.get(str(x), str(x)) for x in adata.obs["sample"]])

    distance = np.sqrt((((x_gaga - x_gaga.mean(0)) / x_gaga.std(0)) ** 2).sum(1))
    keep = np.ones(len(x_gaga), dtype=bool)
    keep[np.argsort(distance)[-N_PLOT_OUTLIERS:]] = False
    x_plot = x_gaga[keep]

    figure = plt.figure(figsize=(9, 7), facecolor="white")
    axis = figure.add_subplot(111, projection="3d", facecolor="white")
    handles = []
    for name, color in sample_colors.items():
        points = x_plot[sample[keep] == name]
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=5, alpha=0.75,
                     rasterized=True, depthshade=True, linewidths=0)
        handles.append(mlines.Line2D([], [], marker="o", color="w", markerfacecolor=color,
                                    markersize=7, label=name))
    finalise_3d(axis)
    axis.legend(handles=handles, **LEGEND_KW)
    figure.savefig(FIGURE_DIR / "Figure7b_samples.png", dpi=250, bbox_inches="tight")
    plt.close(figure)

    starts = np.concatenate([t1[:, 0, :], t2[:, 0, :]])
    start_cells = np.unique(
        np.argsort(pairwise_distances(starts, x_gaga, n_jobs=-1), axis=1)[:, :5]
    )
    t1_end_cells = np.unique(
        np.argsort(pairwise_distances(t1[:, -1, :], x_gaga, n_jobs=-1), axis=1)[:, :5]
    )
    n_per_cluster = np.array([(labels == value).sum() for value in range(3)])
    max_n = n_per_cluster.max()
    top_k = {
        value: max(5, int(round(5 * max_n / n_per_cluster[value])))
        for value in range(3)
    }
    endpoint_cells = {
        value: np.unique(
            np.argsort(
                pairwise_distances(t2[labels == value, -1, :], x_gaga, n_jobs=-1),
                axis=1,
            )[:, :top_k[value]]
        )
        for value in range(3)
    }
    cell_group = np.array(["other"] * adata.n_obs, dtype=object)
    cell_group[start_cells] = "start"
    cell_group[t1_end_cells] = "T1_end"
    cell_group[endpoint_cells[1]] = "not_shown"
    cell_group[endpoint_cells[0]] = "not_shown"
    cell_group[endpoint_cells[T2_LABEL]] = "T2_end"
    groups = ("other", "start", "T1_end", "T2_end")
    points = {name: x_plot[cell_group[keep] == name] for name in groups}
    t1_draw = evenly_spaced(np.where(t1_mask)[0])
    t2_draw = evenly_spaced(np.where(t2_mask & (labels == T2_LABEL))[0])

    figure = plt.figure(figsize=(9, 7), facecolor="white")
    axis = figure.add_subplot(111, projection="3d", facecolor="white")
    for name in groups:
        value = points[name]
        axis.scatter(
            value[:, 0], value[:, 1], value[:, 2], c=CLUSTER_COLORS[name],
            s=CLUSTER_SIZE[name], alpha=CLUSTER_ALPHA[name], rasterized=True,
            depthshade=True, linewidths=0, zorder=1,
        )
    for i in t1_draw:
        axis.plot(*t1[i].T, color=T1_COLOR, alpha=0.7, lw=1.2)
    for i in t2_draw:
        axis.plot(*t2[i].T, color=T2_COLOR, alpha=0.7, lw=1.2)
    finalise_3d(axis)
    scatter_handles = [
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor=CLUSTER_COLORS[name],
                      markersize=6, label=label)
        for name, label in (("start", "start"), ("T1_end", "T1"), ("T2_end", "T2"))
    ]
    axis.legend(handles=scatter_handles, **LEGEND_KW)
    figure.savefig(FIGURE_DIR / "Figure7c_growth_filtered_trajectories.png", dpi=250,
                   bbox_inches="tight")
    plt.close(figure)

    for name, trajectory, draw, end_name, color in (
        ("T1", t1, t1_draw, "T1_end", T1_COLOR),
        ("T2", t2, t2_draw, "T2_end", T2_COLOR),
    ):
        figure = plt.figure(figsize=(9, 7), facecolor="white")
        axis = figure.add_subplot(111, projection="3d", facecolor="white")
        for group, point_color in (("start", "#9E9E9E"), (end_name, "#000000")):
            value = points[group]
            axis.scatter(value[:, 0], value[:, 1], value[:, 2], c=point_color, s=10,
                         alpha=0.9, rasterized=True, depthshade=True, linewidths=0)
        for i in draw:
            axis.plot(*trajectory[i].T, color=color, alpha=0.7, lw=1.2)
        finalise_3d(axis)
        axis.legend(
            handles=[
                mlines.Line2D([], [], marker="o", color="w", markerfacecolor="#9E9E9E",
                              markersize=7, label="Start"),
                mlines.Line2D([], [], marker="o", color="w", markerfacecolor="#000000",
                              markersize=7, label="End"),
                mlines.Line2D([], [], color=color, lw=1.6, label=name),
            ],
            **LEGEND_KW,
        )
        figure.savefig(FIGURE_DIR / f"Figure7d_{name}_trajectory.png", dpi=250,
                       bbox_inches="tight")
        plt.close(figure)


def plot_figure7f(name, gene_file, adata, t1, t2, model, scaler, pcs, means):
    genes = [line.strip() for line in gene_file.read_text().splitlines() if line.strip()]
    lookup = {str(gene): i for i, gene in enumerate(adata.var_names)}
    missing = [gene for gene in genes if gene not in lookup]
    if missing:
        raise KeyError(f"Figure 7f genes missing from H5AD: {missing}")
    hvg = np.asarray(adata.var["highly_variable"], dtype=bool)
    not_hvg = [gene for gene in genes if not hvg[lookup[gene]]]
    if not_hvg:
        raise ValueError(f"Figure 7f genes outside the PCA gene set cannot be decoded: {not_hvg}")
    indices = np.array([lookup[gene] for gene in genes])
    values = {
        "T1": decode_selected(t1, indices, model, scaler, pcs, means),
        "T2": decode_selected(t2, indices, model, scaler, pcs, means),
    }
    columns = 5
    rows = int(math.ceil(len(genes) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(2.0 * columns, 1.9 * rows), squeeze=False)
    for column, gene in enumerate(genes):
        axis = axes.flat[column]
        for group, color in (("T1", T1_COLOR), ("T2", T2_COLOR)):
            mean = values[group][:, :, column].mean(0)
            std = values[group][:, :, column].std(0)
            axis.plot(mean, color=color, lw=1.1)
            axis.fill_between(np.arange(len(mean)), mean - std, mean + std,
                              color=color, alpha=0.18, linewidth=0)
        axis.set_title(gene, fontsize=11, pad=3)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.flat[len(genes):]:
        axis.axis("off")
    figure.legend(
        handles=[mlines.Line2D([], [], color=T1_COLOR, lw=1.6, label="T1"),
                 mlines.Line2D([], [], color=T2_COLOR, lw=1.6, label="T2")],
        loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0),
    )
    figure.tight_layout(h_pad=0.6, w_pad=0.6, rect=(0, 0, 1, 0.95))
    figure.savefig(FIGURE_DIR / f"Figure7f_{name}_gene_trends.png", dpi=300)
    figure.savefig(FIGURE_DIR / f"Figure7f_{name}_gene_trends.pdf", dpi=300)
    plt.close(figure)
    return genes


def compute_effect_sizes(adata, t1, t2, model, scaler, pcs, means):
    keep = qc_gene_mask(adata)
    keep_indices = np.where(keep)[0]
    genes = adata.var_names[keep].astype(str).to_numpy()

    def mean_profile(latent):
        with torch.no_grad():
            pca_scaled = model.decode(torch.tensor(latent.reshape(-1, 3), dtype=torch.float32)).numpy()
        pca = scaler.inverse_transform(pca_scaled).reshape(latent.shape[0], latent.shape[1], -1)
        return pca.mean(0) @ pcs[keep_indices].T + means[keep_indices]

    informative = np.abs(mean_profile(t2) - mean_profile(t1)).mean(0) > 0
    selected = keep_indices[informative]
    selected_genes = genes[informative]
    tables = []
    for start in range(0, len(selected), 512):
        stop = min(start + 512, len(selected))
        indices = selected[start:stop]
        interest = decode_selected(t2, indices, model, scaler, pcs, means)
        baseline = decode_selected(t1, indices, model, scaler, pcs, means)
        tables.append(
            effect_size_table(selected_genes[start:stop], interest, baseline)
        )
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(OUT_DIR / "per_gene_effectsize.csv", index=False)
    return {
        "n_qc_genes": int(keep.sum()),
        "n_informative_genes": int(informative.sum()),
        "n_T1_trajectories": int(t1.shape[0]),
        "n_T2_trajectories": int(t2.shape[0]),
    }


def main() -> None:
    GRANGER_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(IN_DIR / "in_vivo.h5ad")
    if "is_tf" not in adata.var or "highly_variable" not in adata.var:
        raise KeyError("in_vivo.h5ad requires var['is_tf'] and var['highly_variable']")
    model, scaler, pcs, means = load_decoder(adata)
    t1_all = np.load(TRAJ_DIR / "T1_trajectories_latent.npy")
    t2_all = np.load(TRAJ_DIR / "T2_trajectories_latent.npy")
    labels = np.load(TRAJ_DIR / "T2_endpoint_labels.npy")
    t1_mask = np.load(FILTER_DIR / "T1_to_grow_mask.npy")
    t2_mask = np.load(FILTER_DIR / "T2_to_grow_mask.npy")
    t1 = t1_all[t1_mask]
    t2 = t2_all[t2_mask & (labels == T2_LABEL)]

    hvg = adata.var_names[np.asarray(adata.var["highly_variable"], dtype=bool)].astype(str).tolist()
    tfs = adata.var_names[np.asarray(adata.var["is_tf"], dtype=bool)].astype(str).tolist()
    union = set(hvg) | set(tfs)
    gene_indices = np.array([i for i, gene in enumerate(adata.var_names.astype(str)) if gene in union])
    genes = adata.var_names[gene_indices].astype(str).to_numpy()
    decoded = {
        "T1": decode_selected(t1, gene_indices, model, scaler, pcs, means),
        "T2": decode_selected(t2, gene_indices, model, scaler, pcs, means),
    }
    for name, k in (("T1", 4), ("T2", 3)):
        clusters, trends = cluster_trends(decoded[name], genes, k)
        clusters.to_csv(OUT_DIR / f"{name}_temporal_clusters.csv")
        trends.to_csv(OUT_DIR / f"{name}_trends.csv")
        run_granger_group(name, decoded[name], genes, hvg, tfs)

    x_gaga = np.load(TRAJ_DIR / "X_gaga.npy")
    plot_figure7bcd(adata, x_gaga, t1_all, t2_all, t1_mask, t2_mask, labels)
    figure7f_genes = {
        name: plot_figure7f(name, ROOT / "config" / f"figure7f_{name}_genes.txt",
                            adata, t1, t2, model, scaler, pcs, means)
        for name in ("T1", "T2")
    }
    effect_report = compute_effect_sizes(adata, t1, t2, model, scaler, pcs, means)
    summary = {
        "decoder": "GAGA decoder applied to latent trajectories, then PCA inverse transform",
        "growth_filtered_shapes": {"T1": list(t1.shape), "T2": list(t2.shape)},
        "effect_size": effect_report,
        "figure7f_genes": figure7f_genes,
    }
    write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
