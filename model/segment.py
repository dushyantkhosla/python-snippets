import pandas as pd
from pandas import Series, DataFrame
from sklearn.cluster import MiniBatchKMeans, KMeans
from hdbscan import HDBSCAN

def autoCluster(X, krange=[6, 8, 12, 16, 20], method='MiniBatch'):
    """
    Run KMeans/Minibatch on PCA Output for different values of 'k' and find SS

    Parameters
    ----------
    X: DataFrame or np.ndarry
        input data, or possibly the output from autoPCA

    krange: list
        Possible values of k to iterate over and find the one with lowest SS

    method: str
        Which clustering algorithm to use. default 'MiniBatch'

    Returns
    -------
    dict_clus: dict
        containing,
        Sum of Squares: Series
        Clustering Labels: DataFrame

    """
    print("Running Cluster solutions ... ")

    ss = {}
    labels = {}

    for k in krange:
        print(k, end=', '),
        if method=='MiniBatch':
            mb = MiniBatchKMeans(n_clusters=k, random_state=1123456)
        else:
            mb = KMeans(n_clusters=k, random_state=123456)

        labels['mb_' + str(k).zfill(2)] = mb.fit_predict(X)
        ss['Clus_' + str(k).zfill(2)] = mb.inertia_

    _ = Series(ss, index=ss.keys()).plot(
        figsize=(16, 4),
        xticks=range(len(krange))
    )
    _.set_ylabel("Sum of Squares\n")
    _.set_xlabel("\n Number of Clusters (k)")


    dict_clus = {
        'sum_of_squares': Series(ss),
        'labels':DataFrame(labels)
    }

    return dict_clus

def explore_clusters_vs_y(dfclus, y):
    """
    Find the cluster number and size
    of the cluster with highest mean_y

    Parameters
    ----------
    dfclus: pandas.DataFrame
        A DataFrame containing all cluster variables

    y: pandas.Series
        The dependent variable

    Returns
    -------
    df: pandas.DataFrame
        For each Cluster solution, lists Clus, y_mean and size.

    """
    import pandas as pd
    df_ = pd.concat([dfclus, y], axis=1)

    s_ = df_.apply(lambda c: c.value_counts(normalize=True)).unstack().dropna()
    s_.name = 'size'
    s_.index.names = ['index', 'idxmax']

    df = \
    (df_
     .apply(lambda c: df_.groupby(c)['y'].mean())
     .agg(['max', 'idxmax'])
     .T
     .drop('y')
     .reset_index()
     .set_index(['index', 'idxmax'])
     .join(s_.to_frame())
     .round(2)
     .reset_index()
     .rename(columns={'index': 'K', 'idxmax': 'Clus', 'max': 'y_mean'})
    )

    return df



def run_hdbscan(X, COLS, MCS, MS, DF, y):
    """
    Run hdbscan for given data and combination of Min Cluster Size, Min Samples

    Parameters
    ----------
    X: pandas.DataFrame
        The input data to be clustered

    COLS: list
        The columns from X to be considered

    MCS: int
        Min Cluster Size, input to HDBSCAN()

    MS: int
        Min Samples, input to HDBSCAN()

    DF: pandas.DataFrame
        Data for profiling

    Returns
    -------
    Dict with the profile and labels
    """
    hdb = \
    (HDBSCAN(
        min_cluster_size=MCS,
        min_samples=MS)
     .fit(X[COLS])
    )

    df = \
    (DF
     .join(y)
     .assign(clus = hdb.labels_)
     .query("clus != -1")
    )

    profile = \
    (df
     .groupby('clus')
     .mean()
     .join(Series(hdb.labels_, name='size').value_counts(normalize=True))
     .T
     .round(2)
     .loc[COLS + ['size', 'y'], :]
    )

    return {
        'profile': profile.loc[['size', 'y']].T.sort_values('y').query("y > 0.2"),
        'labels': hdb.labels_
    }
