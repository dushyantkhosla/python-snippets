def autoPCA(df, threshold=0.85, lookup=[40, 60, 80, 100], clip=True, clip_at=0.99):
    """
    Find the n_components of a PCA that explain 85% of the variance

    Parameters
    ----------
    df: pandas.DataFrame
        independent variables

    threshold: float
        explained variance should be >= threshold

    lookup: list (of ints)
        potential n_components values to search over

    clip: bool
        if True, input variables will be clipped

    clip_at: float
        values higher than the clip_at percentile will be clipped

    Returns
    -------
    dict of PCA results

    Usage
    -----
    autoPCA(df=basetable, lookup=range(50, 55), thresh=0.9)

    """
    import time
    import pandas as pd
    from pandas import Series, DataFrame
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    dict_pca = {}

    print("\nImputing and Scaling the data...")
    X_scaled = StandardScaler().fit_transform(df)

    if clip == True:
        print("Clipping at the {}th Percentile to control outliers...".format(clip_at * 100))
        df = df.apply(lambda x: x.clip_upper(x.quantile(clip_at)))

    t0 = time.time()
    print("\nRunning PCA for different values of n_components provided...")

    for n in lookup:
        print(n,)
        pca_n = PCA(n_components=n, whiten=True)
        pca_n.fit(X_scaled)
        dict_pca[n] = round(pca_n.explained_variance_ratio_.sum(), 2)


    print("\nFinished running PCA in {} seconds.".format(round(time.time() - t0, 2)))

    # Explained Variance Ratio
    evr = Series(dict_pca)
    evr.plot(title = "Explained Variance Ratio vs. n_components")

    try:
        n_comp = evr[evr > threshold].head(1).index[0]
        evr_ = 100 * evr[evr>threshold].head(1).values[0]
        print("\nSelected the {} Component Solution explaining {}% of the Variance.".format(n_comp, evr_))

        print("Fitting and Transforming input data to {} Components...".format(n_comp))
        pca_n = PCA(n_components=n_comp, whiten=True)
        X_pca = pca_n.fit_transform(X_scaled)

        print("Creating a DataFrame of Factor Loadings")
        pca_loadings = DataFrame(
            pca_n.components_,
            columns=df.columns,
            index=['Comp_' + str(x) for x in range(1, 1 + n_comp)])

        print("Process Complete. \nReturning 'n_comp', 'pca_loadings', and 'pca_output'  in a dict.")
        return {'n_comp': n_comp,
                'pca_loadings': pca_loadings.T.round(2),
                'pca_output': X_pca}

    except:
        print("\nError! Provided threshold not found. Try widening the lookup range.")
        return None
