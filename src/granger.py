"""Pairwise Granger causality."""
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tsa.stattools import grangercausalitytests
from tqdm import tqdm

maxlag = (1,)
test = "ssr_chi2test"

def grangers_causation_matrix(
    data, in_variables, out_variables, test="ssr_chi2test", n_jobs=1, warn=False
):
    def get_pval(dd):
        with warnings.catch_warnings():
            if not warn:
                warnings.simplefilter("ignore", category=FutureWarning)
            test_result = grangercausalitytests(dd, maxlag=maxlag, verbose=False)

        p_values = [test_result[i][0][test][1] for i in maxlag]
        coefs = [test_result[i][1][1].params[1] for i in maxlag]

        arg_min_p_value = np.argmin(p_values)
        min_p_value = p_values[arg_min_p_value]
        min_coef = coefs[arg_min_p_value]
        return (min_p_value, min_coef)

    out = Parallel(n_jobs=n_jobs)(
        delayed(get_pval)(data[[c, r]])
        for c in tqdm(out_variables, desc="Granger", disable=not sys.stderr.isatty())
        for r in in_variables
    )
    out_p = [p for (p, c) in out]
    out_c = [c for (p, c) in out]
    shape = (len(out_variables), len(in_variables))
    df_p = pd.DataFrame(np.array(out_p).reshape(shape), columns=in_variables, index=out_variables).T
    df_c = pd.DataFrame(np.array(out_c).reshape(shape), columns=in_variables, index=out_variables).T
    df_p.index = [var + "_x" for var in in_variables]
    df_p.columns = [var + "_y" for var in out_variables]
    df_c.index = [var + "_x" for var in in_variables]
    df_c.columns = [var + "_y" for var in out_variables]
    return df_p, df_c

def do_granger(trajs, in_genes, out_genes, n_jobs=1, warn=False):
    trajs = trajs.T[::10]
    trajs = trajs - trajs.shift(1)
    trajs = trajs.dropna()
    out_traj_p, out_traj_c = grangers_causation_matrix(
        trajs, in_variables=in_genes, out_variables=out_genes, n_jobs=n_jobs, warn=warn
    )
    return out_traj_p, out_traj_c

