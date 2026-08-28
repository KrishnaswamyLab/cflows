# Data

Download the bundle from https://huggingface.co/datasets/xingzhi0/cflows and
unpack it from the repository root (`tar -xzf cflows_data.tar.gz`), so that
`data/in_vitro/` and `data/in_vivo/` exist. `manifest.tsv` lists every file with
its size and SHA-256; `run.py` checks them before running.

Each directory holds one preprocessed H5AD, the fitted model checkpoints and
preprocessing scalers, the in-vitro trajectories, and the in-vivo initial
conditions.
