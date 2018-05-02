import subprocess as sbp
import os

def run_on_bash(command):
    """
    Runs a bash command on the CLI

    Parameters
    ----------
    command: str
        A bash command

    Returns
    -------
    result: str
        The output of running bash command on the CLI
    """
    try:
        result = sbp.check_output("{}".format(command), shell=True).decode("utf-8")
    except:
        print("Error executing '{}'. \nPlease make corrections and try again.".format(command))
        result = None
    return result


def get_file_info(path):
    """
    Returns file size in MB, and number of rows for a file
    Parameters
    ----------
    path: str
        Location of the file
    Returns
    -------
    result: dict
        Size of the file in MB and number of rows
    """
    try:
        if os.path.exists(path):
            rows_ = int(run_on_bash("wc -l {}".format(path)).split(" ")[0])
            size_ = os.path.getsize(path)/10**6
            result = {
                'rows': rows_,
                'size': size_
            }
        else:
            print("File not found")
            result = None
    except:
        print("An error occurred, please check input and try again.")
        result = None
    return result


def create_sample(FILE_, SIZE_):
    """
    Use the UNIX shuf utility to create
    a file with a random selection of rows from a large file

    Parameters
    ----------
    FILE_: str
        The location of the source (large) file

    SIZE_: int
        The number of rows desired in the sample

    Returns
    -------
    None
    """
    FILE_dest = FILE_.replace(".csv", '_sample.csv')

    if os.path.exists(FILE_dest):
        run_on_bash("rm -f {}".format(FILE_dest))

    try:
        if SIZE_ < get_file_info(FILE_).get('rows'):
            print("Selecting a random sample of {} rows".format(SIZE_))
            run_on_bash("head -n 1 {} > {}".format(FILE_, FILE_dest))
            run_on_bash("cat {} | sed '1d' | shuf -n {} >> {}".format(FILE_, SIZE_, FILE_dest))
            print("Sample file created at {}".format(FILE_dest))
        else:
            print("Sampling with replacement...")
            run_on_bash("head -n 1 {} > {}".format(FILE_, FILE_dest))
            run_on_bash("cat {} | sed '1d' | shuf -r -n {} >> {}".format(FILE_, SIZE_, FILE_dest))
            print("Sample file created at {}".format(FILE_dest))
    except:
        print("An error occured. Please check the inputs and try again.")
