"""
Communication statistics script using data from multiple applications.
This script reads all communication matrices in a given directory tree.
For each communication matrix, it computes all communication statistics
using the code available at https://github.com/llpilla/communication-statistics .
It organizes its results in two categories ('comm' & 'page') depending on the
nature of the communication matrix.
"""

import pandas as pd  # for dataframes
from os import walk  # to find files in the directory hierarchy
from communicationstatistics.commstats import CommunicationStatistics

filepath = 'input-ml-explo-matrices'  # directory containing all comm matrices

myfile = 'input-ml-explo-matrices/__cere__blackscholes_m4__Z9bs_threadPv_368/conf1/trace/comm_mat_numalize__cere__blackscholes_m4__Z9bs_threadPv_368_0_16.csv'

def extract_app_and_conf(dirpath):
    """
    Format of 'dirpath':
        {filepath}/{application}/conf[123]/trace
    """
    parts = dirpath.split('/')
    return(parts[1], parts[2])


def extract_type_and_threads(filename):
    """
    Format of a filename:
        [comm|page]_mat_numalize{application}_0_[16|32|64].csv
    """
    parts = filename.split('_')
    return(parts[0], parts[-1][0:2])


def main():
    # Starts the dataframes to be used
    df_columns = ['application', 'conf', 'threads', 'CH', 'CHv2', 'CA',
                  'CB', 'CBv2', 'CC', 'NBC', 'SP(16)']
    page_df = pd.DataFrame(columns=df_columns)
    comm_df = pd.DataFrame(columns=df_columns)
    dfs = {'page': page_df, 'comm': comm_df}

    # Iterates over the directory and sub-directories
    for (dirpath, dirnames, filenames) in walk(filepath):
        # If there are no files in this directory, skips to the next
        if not filenames:
            continue
        # Gets the application name and the configuration number
        application, conf = extract_app_and_conf(dirpath)
        # Iterates over files
        for csvfile in filenames:
            # Gets the type of the matrix and the number of threads used
            matrix_type, threads = extract_type_and_threads(csvfile)
            # Organizes the complete filepath
            completepath = (dirpath + '/' + csvfile)
            print(completepath)
            # Opens the file to compute metrics
            stats = CommunicationStatistics(completepath)
            # Computes metrics
            dfs[matrix_type] = dfs[matrix_type].append({
                'application': application,
                'conf': conf,
                'threads': threads,
                'CH': stats.ch(),
                'CHv2': stats.ch_v2(),
                'CA': stats.ca(),
                'CB': stats.cb(),
                'CBv2': stats.cb_v2(),
                'CC': stats.cc(),
                'NBC': stats.nbc(),
                'SP(16)': stats.sp(16)
            }, ignore_index=True)

    # Saves the dataframes as CSV files
    dfs['comm'].to_csv('statistics-comm.csv', sep=',', index=False)
    dfs['page'].to_csv('statistics-page.csv', sep=',', index=False)

if __name__ == '__main__':
    main()
