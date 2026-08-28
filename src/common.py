from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
EPS = 1e-8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def strip_xy(value: object) -> str:
    text = str(value)
    return text[:-2] if text.endswith(("_x", "_y")) else text


def n_jobs() -> int:
    return int(os.environ.get("CFLOWS_N_JOBS", "-1"))


def save_granger(prefix: Path, pvals: pd.DataFrame, coefs: pd.DataFrame) -> pd.DataFrame:
    signed = (-np.log(pvals + 2**-10)) * np.sign(coefs)
    signed.index = [strip_xy(x) for x in signed.index]
    signed.columns = [strip_xy(x) for x in signed.columns]
    pvals.to_csv(prefix.with_name(prefix.name + "_p.csv"))
    coefs.to_csv(prefix.with_name(prefix.name + "_c.csv"))
    signed.to_csv(prefix.with_name(prefix.name + "_signed_score.csv"))
    return signed


def qc_gene_mask(adata) -> np.ndarray:
    names = adata.var_names.astype(str)
    symbol_ok = ~(
        names.str.match(r"^(RPL|RPS)")
        | names.str.match(r"^MT-")
        | names.str.match(r"^[A-Z]{2}[0-9]+\.[0-9]+")
    )
    matrix = adata.X
    if hasattr(matrix, "getnnz"):
        n_cells = np.asarray(matrix.getnnz(axis=0)).ravel()
    else:
        n_cells = np.asarray((matrix > 0).sum(axis=0)).ravel()
    return np.asarray(symbol_ok & (n_cells >= 3))


def effect_size_table(
    genes: np.ndarray, interest: np.ndarray, baseline: np.ndarray
) -> pd.DataFrame:
    n_i, n_b = interest.shape[0], baseline.shape[0]
    mean_i, var_i = interest.mean(0), interest.var(0, ddof=1)
    mean_b, var_b = baseline.mean(0), baseline.var(0, ddof=1)
    pooled = np.sqrt(((n_i - 1) * var_i + (n_b - 1) * var_b) / (n_i + n_b - 2) + EPS)
    cohens_d = ((mean_i - mean_b) / pooled).mean(0)
    welch_t = ((mean_i - mean_b) / np.sqrt(var_i / n_i + var_b / n_b + EPS)).mean(0)
    mean_diff = (mean_i - mean_b).mean(0)
    return pd.DataFrame(
        {"gene": genes, "cohens_d": cohens_d, "welch_t": welch_t, "mean_diff": mean_diff}
    )

