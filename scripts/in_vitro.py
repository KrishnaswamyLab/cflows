#!/usr/bin/env python
"""In vitro: produce Figure 5a-c."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib import gridspec
from scipy import stats
from scipy.signal import argrelmin


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import DATA, RESULTS, effect_size_table, n_jobs, qc_gene_mask, save_granger, write_json  # noqa: E402
from decoder import predict  # noqa: E402
from granger import do_granger  # noqa: E402


IN_DIR = DATA / "in_vitro"
OUT_DIR = RESULTS / "in_vitro"
DECODE_DIR = OUT_DIR / "decode"
GRANGER_DIR = OUT_DIR / "granger"
FIGURE_DIR = OUT_DIR / "figures"
REPRESENTATIVE_GENES = ["ID3", "HMGB2", "FTH1", "SPP1", "MSMP"]
CLUSTER_COLORS = ["#FFCC99", "#FFA07A", "#FF8C00", "#FF4500", "#8B2500"]


def gene_mean(adata) -> np.ndarray:
    if "pca_gene_mean" not in adata.var:
        raise KeyError("in_vitro.h5ad requires var['pca_gene_mean']")
    return np.asarray(adata.var["pca_gene_mean"], dtype=np.float32)


def decode_group(label: str, adata) -> Path:
    trajectory = np.load(IN_DIR / f"trajectories_{label}.npy")
    if label.startswith("T"):
        bad = int(trajectory[-1, :, 0].argmin())
        trajectory = trajectory[:, np.arange(trajectory.shape[1]) != bad, :]
    scaler = np.load(IN_DIR / "flow_scaler.npz")
    flat = trajectory.reshape(-1, 2) * scaler["scale"] + scaler["mean"]
    pca = predict(
        flat,
        str(IN_DIR / "decoder.ckpt"),
        str(IN_DIR / "decoder_x_scaler.npz"),
    )
    decoded = pca @ np.asarray(adata.varm["PCs"]).T + gene_mean(adata)
    decoded = decoded.reshape(trajectory.shape[0], trajectory.shape[1], adata.n_vars)
    output = DECODE_DIR / f"traj_gene_space_{label}.npy"
    np.save(output, decoded)
    return output


def run_granger_group(label: str, path: Path, adata) -> None:
    targets = adata.var_names[np.asarray(adata.var["is_target_gene"], dtype=bool)].astype(str).tolist()
    tfs = adata.var_names[np.asarray(adata.var["is_tf"], dtype=bool)].astype(str).tolist()
    union = set(targets) | set(tfs)
    mask = np.array([str(gene) in union for gene in adata.var_names])
    names = adata.var_names[mask].astype(str).to_numpy()
    trajectory = np.load(path, mmap_mode="r")[:, :, mask]
    data = np.transpose(trajectory, (1, 0, 2))
    valid = data.mean(axis=0).var(axis=0) != 0.0
    names = names[valid]
    frame = pd.DataFrame(data[:, :, valid].mean(axis=0), columns=names)
    present = set(names)
    tf_kept = [gene for gene in tfs if gene in present]
    target_kept = [gene for gene in targets if gene in present]
    pvals, coefs = do_granger(frame.T, in_genes=tf_kept, out_genes=target_kept, n_jobs=n_jobs())
    save_granger(GRANGER_DIR / f"granger_{label[0]}", pvals, coefs)


def temporal_clusters(path: Path, adata) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = np.asarray(adata.var["is_target_gene"], dtype=bool) | np.asarray(
        adata.var["is_tf"], dtype=bool
    )
    names = adata.var_names[panel].astype(str).to_numpy()
    trajectory = np.load(path, mmap_mode="r")[:, :, panel]
    mean = np.asarray(trajectory.mean(axis=1))
    varying = mean.std(axis=0) != 0
    mean, names = mean[:, varying], names[varying]
    normalized = (mean - mean.min(axis=0)) / (mean.max(axis=0) - mean.min(axis=0))
    trends = pd.DataFrame(normalized.T, index=names, columns=[f"t{i}" for i in range(mean.shape[0])])
    peaks = trends.apply(
        lambda row: np.where(row >= sorted(row)[-1])[0].mean(), axis=1
    )
    sort_order = peaks.argsort()
    trends = trends.iloc[sort_order]
    names = trends.index.to_numpy()
    sorted_peaks = peaks.iloc[sort_order].to_numpy()

    grid = np.linspace(peaks.min(), peaks.max(), 1000)
    density = stats.gaussian_kde(peaks.to_numpy(), bw_method=0.15)(grid)
    minima = argrelmin(density, order=5)[0]
    margin = int(len(grid) * 0.05)
    minima = minima[(minima > margin) & (minima < len(grid) - margin)]
    thresholds = grid[minima]
    if len(thresholds) != 4:
        raise RuntimeError(f"expected four temporal-cluster thresholds; found {len(thresholds)}")
    cuts = [0] + [
        int(np.searchsorted(np.sort(peaks.to_numpy()), value, side="right"))
        for value in thresholds
    ] + [len(trends)]
    cluster = np.empty(len(trends), dtype=int)
    for value, (start, stop) in enumerate(zip(cuts[:-1], cuts[1:])):
        cluster[start:stop] = value
    table = pd.DataFrame(
        {
            "number": np.arange(len(names)),
            "cluster": cluster,
            "peak_index": sorted_peaks,
        },
        index=names,
    )
    table.to_csv(OUT_DIR / "T_temporal_clusters.csv")
    trends.to_csv(OUT_DIR / "T_trends.csv")
    return trends, table


def plot_figure5a(trends: pd.DataFrame, clusters: pd.DataFrame) -> None:
    cmap = mcolors.ListedColormap(sns.color_palette("Oranges", n_colors=5))
    figure = plt.figure(figsize=(5.5, 7))
    layout = gridspec.GridSpec(1, 2, width_ratios=[0.04, 0.96], wspace=0.01)
    strip = figure.add_subplot(layout[0])
    sns.heatmap(clusters[["cluster"]], cmap=cmap, cbar=False, ax=strip,
                yticklabels=False, xticklabels=False)
    heatmap = figure.add_subplot(layout[1])
    sns.heatmap(trends, cmap="Purples", vmin=0, vmax=1, cbar=False, ax=heatmap,
                yticklabels=False)
    heatmap.set_xticks([0, 25, 50, 75, 99])
    heatmap.set_xticklabels(["Day 0", "Day 1", "Day 12", "Day 18", "Day 30"], rotation=90)
    heatmap.set_xlabel("")
    heatmap.set_ylabel("")
    figure.savefig(FIGURE_DIR / "Figure5a_expression_heatmap.pdf", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_figure5b(path: Path, adata, clusters: pd.DataFrame) -> None:
    names = adata.var_names.astype(str).to_numpy()
    index = {gene: i for i, gene in enumerate(names)}
    missing = [gene for gene in REPRESENTATIVE_GENES if gene not in index]
    if missing:
        raise KeyError(f"Figure 5b genes missing from H5AD: {missing}")
    trajectory = np.load(path, mmap_mode="r")[:, :, [index[g] for g in REPRESENTATIVE_GENES]]
    mean, std = trajectory.mean(axis=1), trajectory.std(axis=1)
    figure, axes = plt.subplots(5, 1, figsize=(2.0, 7.2))
    for column, (axis, gene) in enumerate(zip(axes, REPRESENTATIVE_GENES)):
        cluster = int(clusters.loc[gene, "cluster"])
        color = CLUSTER_COLORS[cluster]
        axis.plot(mean[:, column], color=color, lw=1.5)
        axis.fill_between(
            np.arange(mean.shape[0]), mean[:, column] - std[:, column], mean[:, column] + std[:, column],
            color=color, alpha=0.25, linewidth=0,
        )
        axis.set_title(gene, fontstyle="italic", fontsize=11)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.text(-0.28, 0.5, f"C{cluster}", transform=axis.transAxes,
                  ha="right", va="center", fontsize=10, fontweight="bold")
    figure.tight_layout(h_pad=0.7)
    figure.savefig(FIGURE_DIR / "Figure5b_representative_trends.pdf", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_figure5c(clusters: pd.DataFrame) -> None:
    signed = pd.read_csv(GRANGER_DIR / "granger_T_signed_score.csv", index_col=0)
    tfs = [gene for gene in clusters.index if gene in signed.index]
    genes = [gene for gene in clusters.index if gene in signed.columns]
    matrix = signed.loc[tfs, genes]
    tf_cluster = clusters.loc[tfs, "cluster"].astype(int)
    gene_cluster = clusters.loc[genes, "cluster"].astype(int)
    mask = tf_cluster.to_numpy()[:, None] > gene_cluster.to_numpy()[None, :]
    matrix = matrix.copy()
    matrix.values[mask] = 0.0
    matrix.to_csv(GRANGER_DIR / "granger_T_signed_score_temporal_mask.csv")

    orange = mcolors.ListedColormap(sns.color_palette("Oranges", n_colors=5))
    blue = mcolors.ListedColormap(sns.color_palette("Blues", n_colors=5))
    figure = plt.figure(figsize=(10.5, 10.5))
    layout = gridspec.GridSpec(2, 2, width_ratios=[0.035, 0.965], height_ratios=[0.965, 0.035],
                               hspace=0.01, wspace=0.01)
    row_strip = figure.add_subplot(layout[0, 0])
    sns.heatmap(tf_cluster.to_frame(), cmap=orange, cbar=False, ax=row_strip,
                yticklabels=False, xticklabels=False)
    heatmap = figure.add_subplot(layout[0, 1])
    sns.heatmap(matrix, cmap="RdBu_r", center=0, robust=True, cbar=False, ax=heatmap,
                yticklabels=False, xticklabels=False)
    col_strip = figure.add_subplot(layout[1, 1])
    sns.heatmap(gene_cluster.to_frame().T, cmap=blue, cbar=False, ax=col_strip,
                yticklabels=False, xticklabels=False)
    figure.savefig(FIGURE_DIR / "Figure5c_granger_heatmap.png", dpi=300, bbox_inches="tight")
    figure.savefig(FIGURE_DIR / "Figure5c_granger_heatmap.pdf", dpi=300, bbox_inches="tight")
    plt.close(figure)


def compute_effect_sizes(t_path: Path, a_path: Path, adata) -> dict:
    keep = qc_gene_mask(adata)
    genes = adata.var_names[keep].astype(str).to_numpy()
    interest = np.transpose(np.load(t_path, mmap_mode="r")[:, :, keep], (1, 0, 2))
    baseline = np.transpose(np.load(a_path, mmap_mode="r")[:, :, keep], (1, 0, 2))
    informative = np.abs(interest.mean(0) - baseline.mean(0)).mean(0) > 0
    table = effect_size_table(genes[informative], interest[:, :, informative], baseline[:, :, informative])
    table.to_csv(OUT_DIR / "per_gene_effectsize.csv", index=False)
    return {
        "n_qc_genes": int(keep.sum()),
        "n_informative_genes": int(informative.sum()),
        "n_T_trajectories": int(interest.shape[0]),
        "n_A_trajectories": int(baseline.shape[0]),
    }


def main() -> None:
    for directory in (DECODE_DIR, GRANGER_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(IN_DIR / "in_vitro.h5ad")
    required = {"is_target_gene", "is_tf", "pca_gene_mean"}
    if not required.issubset(adata.var.columns):
        raise KeyError(f"in_vitro.h5ad is missing var annotations: {sorted(required - set(adata.var.columns))}")
    a_path = decode_group("A_extreme", adata)
    t_path = decode_group("T_extreme", adata)
    t_full_path = decode_group("T", adata)
    run_granger_group("A_extreme", a_path, adata)
    run_granger_group("T_extreme", t_path, adata)
    trends, clusters = temporal_clusters(t_path, adata)
    plot_figure5a(trends, clusters)
    plot_figure5b(t_full_path, adata, clusters)
    plot_figure5c(clusters)
    effect_report = compute_effect_sizes(t_path, a_path, adata)
    summary = {
        "decode": {
            "A_extreme": list(np.load(a_path, mmap_mode="r").shape),
            "T_extreme": list(np.load(t_path, mmap_mode="r").shape),
            "T": list(np.load(t_full_path, mmap_mode="r").shape),
        },
        "effect_size": effect_report,
        "figure5b_genes": REPRESENTATIVE_GENES,
    }
    write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
