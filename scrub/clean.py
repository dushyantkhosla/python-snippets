import numpy as np
import pandas as pd

def drop_zv(df_):
    """
    Drop columns that have zero-variance
    For Categoricals, if nunique == 1
    For Numeric, if std == 0
    """
    cols_catg_zv = \
    (df_
     .select_dtypes(include='object')
     .nunique()
     .where(lambda i: i == 1)
     .dropna()
     .index
     .tolist()
    )

    cols_numeric_zv = \
    (df_
     .select_dtypes(include=np.number)
     .std()
     .where(lambda i: i == 0)
     .dropna()
     .index
     .tolist()
    )
    
    cols_zv = cols_catg_zv + cols_numeric_zv

    if len(cols_zv) >= 1:
        print("The following columns have zero-variance and will be dropped \n{}".format(cols_zv))
        df_.drop(cols_zv, axis=1, inplace=True)
    else:
        print("No columns with zero-variance.")
    return df_

def drop_nzv(df_, nzv_threshold=0.95):
    """
    """
    cols_catg_nzv = \
    (df_
     .select_dtypes(include='object')
     .apply(lambda c: c.value_counts(normalize=True).agg(['max', 'idxmax']))
     .T
     .query("max > {}".formata(nzv_threshold))
     .index
     .tolist()
    )

    if len(cols_catg_nzv) >= 1:
        print("The mode of these columns has a frequency higher than {}. Dropping these. {}"
              .format(nzv_threshold, cols_catg_nzv))
        df_.drop(cols_catg_nzv, axis=1, inplace=True)
    else:
        print("No categorical columns with near-zero variance found.")

def drop_missings(df_, threshold=0.8):
    """
    """
    cols_missings = \
    (df_
     .isnull()
     .mean()
     .where(lambda i: i > threshold)
     .dropna()
     .index
     .tolist()
    )
    
    if len(cols_missings) >= 1:
        print("The following columns have more than {:.2f}% missings and will be dropped...\n{}"
              .format(threshold * 100, cols_missings))
        df_.drop(cols_missings, inplace=True, axis=1)
    else:
        print("No columns have more than {:.2f}% missings.".format(threshold))
    return df_
