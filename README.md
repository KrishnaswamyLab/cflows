# Cflows

Code and fitted models used for reproducing the Cflows results in Figures 5a–c and 7b–d,f of the
accompanying manuscript (preprint: https://doi.org/10.1101/2023.03.28.534644).

## Quick start

Clone this repository, then from its root:

```bash
wget https://huggingface.co/datasets/xingzhi0/cflows/resolve/main/cflows_data.tar.gz
tar -xzf cflows_data.tar.gz
conda env create -f environment.yml
conda activate cflows
python run.py all
```

`run.py` checks the inputs against `data/manifest.tsv`, then runs the in-vitro
and in-vivo branches, writing to `results/in_vitro/` and `results/in_vivo/`.
Either branch can be run alone with `in-vitro` or `in-vivo` in place of `all`.
`--n-jobs N` limits the Granger workers.

## Tutorial

`tutorial.ipynb` walks through the in-vivo branch: the GAGA latent space,
trajectories from the fitted flow model, the growth filter, decoding to gene
space, and Granger causality. It ships with its outputs, so it can be read
without being run.

```bash
jupyter notebook tutorial.ipynb
```

## License

Non-commercial; see `LICENSE.md`.
