import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from statsmodels.tsa.stattools import grangercausalitytests

lag_order = 1 # since we aggregated the data in to 9 bins we only need 1 lag
maxlag = (
    lag_order,  # becuase we got this value before. We are not suppose to add 1 to it
)
test = "ssr_chi2test"

from joblib import Parallel, delayed

def grangers_causation_matrix(
    data, in_variables, out_variables, test="ssr_chi2test", n_jobs=1, warn=False
):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """

    def get_pval(dd):
        if warn:
            test_result = grangercausalitytests(dd, maxlag=maxlag, verbose=True)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                test_result = grangercausalitytests(dd, maxlag=maxlag, verbose=False)
                # according to the documentation https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.grangercausalitytests.html,
                # the dd has 2 columns, second causes the first.

        p_values = [test_result[i][0][test][1] for i in maxlag] # test_result[i][1] is the unrestricted model, test_result[i][1][0] is the restricted model
        coefs = [test_result[i][1][1].params[1] for i in maxlag] # x1, x2, const

        arg_min_p_value = np.argmin(p_values)
        min_p_value = p_values[arg_min_p_value]
        min_coef = coefs[arg_min_p_value]
        return (min_p_value, min_coef)

    out = Parallel(n_jobs=n_jobs)(
        delayed(get_pval)(data[[c, r]]) # this means r causes c, so r is be in and c is out
        for c in tqdm(out_variables, desc="Processing columns")  # Outer loop progress bar
        for r in in_variables  # Inner loop without progress bar
    )
    out_p = [p for (p,c) in out]
    out_c = [c for (p,c) in out]
    df_p = pd.DataFrame(
        np.array(out_p).reshape((len(out_variables), len(in_variables))), # should be reshaped to len(out_variables), len(in_variables) according to the for loop.
        columns=in_variables,
        index=out_variables,
    ).T # used the correct reshaping, and then transposed the matrix so the x and y are semantically correct (x causes y).
    df_c = pd.DataFrame(
        np.array(out_c).reshape((len(out_variables), len(in_variables))), # should be reshaped to len(out_variables), len(in_variables) according to the for loop.
        columns=in_variables,
        index=out_variables,
    ).T
    df_p.index = [var + "_x" for var in in_variables]
    df_p.columns = [var + "_y" for var in out_variables]
    df_c.index = [var + "_x" for var in in_variables]
    df_c.columns = [var + "_y" for var in out_variables]
    return df_p, df_c

def do_granger(trajs, in_genes, out_genes, n_jobs=1, warn=False):
    # in causes out
    trajs = trajs.T[::10]
    trajs = trajs - trajs.shift(1)
    trajs = trajs.dropna()
    out_traj_p, out_traj_c = grangers_causation_matrix(
        trajs, in_variables=in_genes, out_variables=out_genes, n_jobs=n_jobs, warn=warn
    )
    return out_traj_p, out_traj_c

