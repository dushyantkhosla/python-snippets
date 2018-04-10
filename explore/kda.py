def find_key_drivers(X, y):
    """
    Find variable importance in different ways

    Parameters
    ----------
    data: pandas.DataFrame
        The independent variables, all numeric

    y: pandas.Series
        The dependent variable, boolean

    COLS: list
        Columns to consider

    Returns
    -------
    var_imps: pandas.DataFrame
        Importances (as ranks) via each method for each independent variable

    """
    import numpy as np
    import pandas as pd
    from pandas import Series, DataFrame

    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression

    X_scaled = StandardScaler().fit_transform(X)
    COLS = X.columns

    importance = \
    Series(
        data=X
        .groupby(y)
        .mean()
        .apply(lambda c: 0 if c.loc[1] == 0 else (c.loc[0] - c.loc[1])/float(c.loc[1]))
        .replace([np.inf, -np.inf], np.nan)
        .dropna(),
        index=COLS,
        name='importance'
    ).abs()

    fvals = \
    Series(SelectKBest(k='all', score_func=f_classif).fit(X, y).scores_,
           index=COLS,
           name='f_values').abs()

    pvals = \
    Series(SelectKBest(k='all', score_func=f_classif).fit(X, y).pvalues_,
           index=COLS,
           name='p_values').abs()

    log_coefs = \
    Series(LogisticRegression().fit(X_scaled, y).coef_[0],
           index=COLS,
           name='std_betas').abs()

    svc_coefs = \
    Series(LinearSVC().fit(X_scaled, y).coef_[0],
           index=COLS,
           name='svc_coefs').abs()

    extr_fis = \
    Series(ExtraTreesClassifier().fit(X_scaled, y).feature_importances_,
           index=COLS,
           name='extr_imps').abs()

    rafo_fis = \
    Series(RandomForestClassifier().fit(X_scaled, y).feature_importances_,
           index=COLS,
           name='rafo_imps').abs()

    var_imps = \
    (pd.concat([importance,
                fvals,
                log_coefs,
                svc_coefs,
                extr_fis,
                rafo_fis], axis=1)
     .assign(avg_rank = lambda df: df.apply(lambda c: c.rank()).mean(axis=1))
    )

    return var_imps


def find_key_driversCV(X, y, xCOLS, CV=5):
    """
    Run the find_key_drivers function over subsets of the data
    Average over the results to find robust best predictors

    Parameters
    ----------
    X: pandas.DataFrame
        The input dataFrame, numeric or bool features only

    y: pandas.Series
        The dependent variable

    COLS: list
        Which columns to consider from X, default 'all'

    CV: int
        Number of folds

    Results
    -------
    ser_: pandas.Series
        Averaged ranks of features, indexed by feature name

    """
    import pandas as pd
    from pandas import Series, DataFrame

    from sklearn.model_selection import train_test_split

    list__df_imps = []
    COLS = xCOLS if xCOLS else X.columns

    for run in range(CV):
        X_, _, y_, _ = train_test_split(X.loc[:, COLS], y, test_size=0.1)
        df_imps_ = find_key_drivers(X=X_, y=y_)
        list__df_imps.append(df_imps_)

    df_ranks_ = pd.concat(
        [x.loc[:, 'avg_rank'].astype(int) for x in list__df_imps],
        axis=1
    )

    ser_ = \
    (df_ranks_
     .apply(lambda r: r.mean(), axis=1)
     .astype(int)
     .sort_values()
    )

    return ser_

 
