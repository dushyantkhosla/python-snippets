import numpy as np
import pandas as pd
from pandas import Series, DataFrame

def find_uncorrelated_variables(X, IMPORTANCES, CORR_THRESHOLD=0.75):
    """
    How it works:
    1. Replaces 1s along the diagonal with NaNs.
    2. Replaces all values lower than abs(threshold) with NaNs.
    3. For each column, returns a list of correlated variables  (empty if there are none.)
    4. Pull out the uncorrelated variables into a separate list.
    5. For each group of correlated variables, pick the one with the highest importance.
       Ignore the rest. Repeat.

    Parameters
    ----------
        X: pandas.DataFrame
            The input data

        IMPORTANCES: pandas.DataFrame
            Variable Importances, the output of find_key_drivers()

        CORR_THRESHOLD: float
            Correlation Threshold

    Returns
    -------
    uncorr: list
        List of Uncorrelated variables for modeling
    """
    try:
        df_correlations = X.loc[:, IMPORTANCES.index.tolist()].corr()

        # Find groups of correlated variables
        groups = \
        (df_correlations
         .replace(1, np.nan)
         .round(2)
         .applymap(lambda x: x if np.abs(x) >= CORR_THRESHOLD else np.nan)
         .apply(lambda c: c.dropna().index.tolist())
        )

        # Find variables that aren't correlated with any other variables
        uncorr = \
        (groups
         .map(lambda x: np.nan if bool(x) else x)
         .dropna()
         .index
         .tolist()
        )

        # Retain non-empty groups
        corr_groups = \
        (groups
         .map(lambda e: e if bool(e) else np.nan)
         .dropna()
         .to_dict()
        )

        corr_groups = Series([corr_groups[k] + [k] for k in corr_groups])

        ignored = []
        for i in corr_groups:
            """
            """
            options = list(set(i) - set(uncorr + ignored))
            if bool(options):
                selected = IMPORTANCES.loc[options].idxmax()
                uncorr.append(selected)
                options.remove(selected)
                ignored.extend([x for x in options if x != selected])

        print("Selected {} Uncorrelated Variables.".format(len(uncorr)))
        print("Here are a few examples...\n")
        print(Series(sorted(uncorr)).sample(15).tolist())
        return uncorr
    except:
        print("Oops, something went wrong. Please try again.")
