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
