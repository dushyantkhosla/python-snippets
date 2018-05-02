import os
import sys
import time
import string
import trans
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import seaborn as sns

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    from src.utils import time_my_func

# -----------------------------------
# Define helper objects
# -----------------------------------

errors_func_1 = []
errors_func_2 = []
errors_func_3 = []
# ...

dict_replace_1 = {}
dict_replace_2 = {}
dict_replace_3 = {}
# ...

def scrub_col_X():
    """
    """
    pass

def scrub_col_Y():
    """
    """
    pass

def scrub_col_Z():
    """
    """
    pass
# ...

# -----------------------------------
# Define processing functions
# -----------------------------------
@time_my_func
def engineer_features(df):
    """
    Contains code for
    * creating,
    * mutating,
    * clippping, and
    * removing
    columns from the passed df
    """
    df_ = df.copy()
    srs_1 = Series()
    srs_2 = Series()

    df_.loc[:, 'newcol_1'] = \
    (srs_1
    .map(lambda i: )
    .replace()
    .pipe(clip_categorical)
    )

    df_.loc[:, 'newcol_2']
    df_.loc[:, 'newcol_3']
    # ....

    df_.drop(['col_x', 'col_y'],axis=1, inplace=True)
    return df_

def aggregate_grouped(gdf):
    """
    Reduces a subset of the df to one-row-per-entity

    Parameters
    ----------
    gdf: DataFrame
        The grouped (subset) DataFrame passed by groupby().apply()

    Returns
    -------
        Series object containing reductions
    """
    df_ = gdf.copy()
    key = ''
    try:
        s1 = df_.loc[:, ]
        s2 = df_.loc[:, ]
        s3 = Series({
            'k1': df_.loc[:, ''].nunique(),
            'k2': df_.loc[:, ''].dt.strftime("%b_%Y").nunique()
            # ...
        })

        return pd.concat([s1, s2, s3])
    except:
        errors_aggregation.append(key)


# -----------------------------------
# Execute scrubbing
# -----------------------------------

if __name__ == '__main__':
    sys.path.append(os.getcwd())

    from src.obtain import run_on_bash, get_file_info
    from src.obtain import import_filter_df, get_target_df, backup_df
    from src.scrub import drop_zv, drop_nzv, drop_missings, make_dummies, clip_categorical
    from src.scrub import compress_numeric
    from src.obtain import connect_to_db, load_file_to_db, print_table_names

    path_raw = "data/raw/gravity_activity_20180406.csv"
    path_clean = "data/processed/clean_activity.csv"
    path_clean_db = "data/interim/clean.db"

    if os.path.exists(path_clean):
        print("Cleaned file available at {}".format(path_clean))
    else:
        try:
            print("\nBeginning data import")
            t0 = time.time()
            print("--------------------------")

            df_raw = import_filter_df(path_raw)

            print("\nImport Complete.")
            print("Took {:.2f} minutes.".format((time.time()-t0)/60))
            print("--------------------------")
        except:
            print("Import and filter failed.")

        try:
            print("\nBeginning data cleaning...")
            t0 = time.time()
            print("--------------------------")

            df_scrubbed = \
            (df_raw
            .pipe(remove_zv_missings)
            .pipe(engineer_features)
            .pipe(create_dummified_df)
            )

            print("\nCleaning done.")
            print("Took {:.2f} minutes.".format((time.time()-t0)/60))
            print("--------------------------")
        except:
            print("Feature engineering failed.")

        try:
            print("\nAggregating data")
            t0 = time.time()
            print("--------------------------")
            print("This will take a while, go grab a coffee")

            errors_aggregation = []

            df_aggregated = \
            (df_scrubbed
            .groupby('')
            .apply(aggregate_grouped)
            )

            print("Aggregation complete.")
            print("Took {:.2f} minutes.".format((time.time()-t0)/60))
            print("Dataset has {} rows and {} columns\n"
                  .format(df_clean.shape[0], df_clean.shape[1]))
            print("--------------------------")

            if len(errors) >= 1:
                print("{} errors ignored during aggregation.".format(len(errors)))
        except:
            print("Aggregation failed")

        try:
            print("\nStarting backup")
            t0 = time.time()
            print("--------------------------")

            backUp_aggregated_df(df=df_clean, path_clean=path_clean)

            tbl_ = path_clean.split('/')[-1].replace('clean_', '').replace('.csv', '').strip()
            load_file_to_db(path_to_file=path_clean,
                    path_to_db=path_clean_db,
                    table_name=tbl_,
                    delim=',')

            print("Took {:.2f} minutes.".format((time.time()-t0)/60))
            print("File saved at {}, and table created in {} by the name {}"
            .format(path_clean, path_clean_db, tbl_))
            print("\nOkay, All done! Happy exploring!")
            print("--------------------------")
        except:
            print("Backup failed")
