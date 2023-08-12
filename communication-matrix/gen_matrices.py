#!/usr/bin/env python3
import argparse
import collections
import os
import re
import sys
import numpy as np
import pandas as pd

class Gen:

    class Skip(Exception):
        pass

    def __init__(self, args):
        self.COMM_MAT_PREFIX = 'comm_mat_'
        self.SHARED_PAGE_MAT_PREFIX = 'page_mat_'
        self.args = args

    def _log(self, msg):
            if self.args.parallel:
                print(f'{os.getpid()}| {msg}', file=sys.stderr)
            else:
                print(msg, file=sys.stderr)

    def _get_nb_threads(self, input_filename):
        # thread number extraction
        if not input_filename.endswith('.csv'):
            self.log(f'skip {input_filename} without .csv suffix')
            raise self.Skip()

        # input_filename validation
        if self.args.threads is None:
            m = re.match(r'.+_(\d+)\.csv', input_filename)
            if m is None:
                self._log(f'skip {input_filename} without number of threads')
                raise self.Skip()
            nb_threads = int(m.group(1))
        else:
            nb_threads = int(self.args.threads)
        return nb_threads

    def _get_output_filename(self, input_filename, prefix):
        # output_filename generation and sanity check
        if self.args.dir is None:
            output_dirname = os.path.dirname(input_filename)
        else:
            output_dirname = self.args.dir
        output_filename = os.path.join(output_dirname, prefix+os.path.basename(input_filename))
        if output_filename == input_filename:
            self._log(f'skip identical input/output "{input_filename}"')
            raise self.Skip()
        return output_filename

    def _gen_comm_matrix(self, df, nb_threads, output_filename):
        # drop unnecessary column, original df is unchanged
        df = df.drop(columns='page.address')
        df.columns = pd.RangeIndex(nb_threads)

        # initialize the matrix
        mat = np.zeros((nb_threads, nb_threads), np.int64)

        # compute comms with peer threads
        for i in range(nb_threads):
            mat[i][  :i] = df[df[i] > 0].iloc[:,  :i].sum().astype(np.int64)
            mat[i][i+1:] = df[df[i] > 0].iloc[:,i+1:].sum().astype(np.int64)

        # sum the matrix with its transpose
        mat = mat + mat.T

        # write the matrix as a CSV file
        np.savetxt(output_filename, mat, delimiter=',', fmt='%d')

    def _gen_shared_page_matrix(self, df, nb_threads, output_filename):
        df = df.groupby('page.address').sum() # merge common page rows
        df = df.applymap(lambda x: 1 if x>0 else 0)
        df.columns = pd.RangeIndex(nb_threads)

        # initialize the matrix
        mat = np.zeros((nb_threads, nb_threads), np.int64)

        # compute comms with peer threads
        for i in range(nb_threads):
            mat[i][  :i] = df[df[i] > 0].iloc[:,  :i].sum().astype(np.int64)
            mat[i][i+1:] = df[df[i] > 0].iloc[:,i+1:].sum().astype(np.int64)

        # write the matrix as a CSV file
        np.savetxt(output_filename, mat, delimiter=',', fmt='%d')

    def process_file(self, input_filename):
        try:
            nb_threads = self._get_nb_threads(input_filename)
            comm_mat_filename = self._get_output_filename(input_filename, self.COMM_MAT_PREFIX)
            shared_page_mat_filename = self._get_output_filename(input_filename, self.SHARED_PAGE_MAT_PREFIX)

            # log operation
            if self.args.verbose:
                self._log(f'{input_filename} --> {comm_mat_filename} | {shared_page_mat_filename}: {nb_threads} threads')

            # read the CSV trace, load only the per-thread access columns
            labels = ['page.address'] + [f'T{i}' for i in range(nb_threads)]
            df = pd.read_csv(input_filename, sep=',', header=0, usecols=labels, dtype=np.int64)
            self._gen_comm_matrix(df, nb_threads, comm_mat_filename)
            self._gen_shared_page_matrix(df, nb_threads, shared_page_mat_filename)


        except self.Skip:
            pass

def main():
    arg_parser = argparse.ArgumentParser(description='Generate thread communication matrices and shared pages matrices from Numalize traces.')
    arg_parser.add_argument('-d', '--dir', help='set target DIRECTORY to store matrix files (default: same as trace directory)', metavar='DIR')
    arg_parser.add_argument('-t', '--threads', help='assume THREADS number of threads (default: obtained from trace filename)', metavar='THREADS')
    arg_parser.add_argument('-j', '--parallel', help='enable multiprocessing', action="store_true")
    arg_parser.add_argument('-v', '--verbose', help='enable verbose output', action="store_true")
    arg_parser.add_argument('file_list', help='trace filename', nargs='*', metavar='TRACE')

    args = arg_parser.parse_args()
    file_list = args.file_list
    del args.file_list

    gen = Gen(args)
    if args.parallel:
        if args.verbose:
            print('parallel processing enabled', file=sys.stderr)
        from multiprocessing import Pool
        with Pool() as pool:
            collections.deque(pool.map(gen.process_file, file_list), 0)
    else:
        collections.deque(map(gen.process_file, file_list), 0)

if __name__ == '__main__':
    main()
