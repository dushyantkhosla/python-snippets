import os
import sqlite3
import pandas as pd
import numpy as np

from obtain import get_file_info

def connect_to_db(path):
    """
    Interact with a SQLite database

    Parameters
    ----------
    path: str
        Location of the SQLite database

    Returns
    -------
    conn: Connector
        The SQLite connection object

    curs: Cursor
        The SQLite cursor object

    Usage
    -----
    conn, curs = connect_to_db("data/raw/foo.db")
    """
    try:
        if os.path.exists(path):
            print("Connecting to Existing DB")
            conn = sqlite3.connect(path)
        else:
            print("Initialising new SQLite DB")
            conn = sqlite3.connect(path)
        curs = conn.cursor()
    except:
        print("An error occured. Please check the file path")
    return conn, curs

def print_table_names(path_to_db):
    """
    Print and return the names of tables in a SQLite database
    """
    conn, curs = connect_to_db(path_to_db)
    result = curs.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    print(result)
    return result


def load_file_to_db(path_to_file, path_to_db, table_name, delim):
    """
    Load a text file of any size into a SQLite database

    Parameters
    ----------
    path_to_file: str
        Location of the text file
    path_to_db: str
        Location of the SQLite db
    table_name: str
        Name of the table to be created in the database
    delim: str
        The delimiter for the text file

    Returns
    -------
    None
    """
    conn, curs = connect_to_db(path_to_db)
    print("The database at {} contains the following tables.".format(path_to_db))
    print(curs.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
    if os.path.exists(path_to_file):
        size_ = get_file_info(path_to_file).get('size')
        rows_ = get_file_info(path_to_file).get('rows')
        try:
            if size_ < 250:
                print("{} is a small file. Importing directly.".format(path_to_file))
                df_ = pd.read_csv(
                    path_to_file,
                    sep=delim,
                    low_memory=False,
                    error_bad_lines=False,
                    quoting=csv.QUOTE_NONE
                    )

                df_.to_sql(
                    name=table_name,
                    con=conn,
                    index=False,
                    if_exists='append'
                    )
                print("Done.")
            else:
                print("{} is large. Importing in chunks.".format(path_to_file))
                csize = int(np.ceil(rows_/10))
                chunks = pd.read_csv(
                    path_to_file,
                    sep=delim,
                    chunksize=csize,
                    error_bad_lines=False,
                    low_memory=False,
                    quoting=csv.QUOTE_NONE
                    )
                for c in chunks:
                    c.to_sql(
                        name=table_name,
                        con=conn,
                        index=False,
                        if_exists='append'
                        )
                print("Done")
        except:
            print("An error occurred while reading the file.")
    else:
        print("File not found at {}, please check the path".format(path_to_file))
    return None
