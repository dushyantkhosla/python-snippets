import pandas as pd

def make_dummies(ser, DROP_ONE=True):
    """
    Create dummies for different levels of a clipped categorical
    Drop one to avoid the trap

    Parameters
    ----------
    ser: input categorical series
        pandas.Series

    Returns
    -------
    df_dum: dummy variables with one level dropped
        pandas.DataFrame

    """

    if ser.nunique() > 10:
        print("Categorical has too many levels, consider clipping")
        df_dum = None
    else:
        df_dum = pd.get_dummies(ser, prefix=ser.name)
        if DROP_ONE:
            other_col = [c for c in df_dum if 'Other' in c]
            to_drop_ = other_col if other_col else df_dum.mean().idxmin()
            print("Dropping {}\n".format(to_drop_))
            df_dum.drop(to_drop_, axis=1, inplace=True)

    return df_dum
