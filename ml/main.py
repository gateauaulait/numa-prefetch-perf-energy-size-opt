#!/usr/bin/env python3

import sys
import csv
import os
import random
import time
import numpy as np
import subprocess
from subprocess import call
from subprocess import Popen, PIPE
import math
from shutil import copyfile

from sklearn.cluster import KMeans
from sklearn.cluster import ward_tree
from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin_min
from pyeasyga import pyeasyga
import os.path

import pickle

import settings
import statistics

from read_data import read_info
from study import get_optimal_gains
from study import subset_input
from study import subset_labels
from study import project_space
from classify import call_classify
from ga_explo import exlpore_GA

def main():
    if len(sys.argv) != 3:
        print("Incorrect execution usage. Please Try:")
        print("python3 main.py 2 energy")
        print("python3 main.py 1 time_med")
        sys.exit()
    settings.init_data()
    
    settings.GOAL_INPUT = str(sys.argv[1])
    settings.GOAL_TYPE = str(sys.argv[2])

    read_info()     

    # Optional -- this does not affect the data but is left for regression test
    # This command depends on the arguments passed:
    # python3 main.py 2 energy - will explore the gains of size large energy optimization
    # sub_th,sub_node,sub_pref,sub_hyper,sub_th_map,sub_data
    # subset_labels(32,2,0,0,"scatter","ori")        
    R = subset_labels(settings.LABELS,settings.CONFIGURATIONS_LABELS,"","","","","","")
    settings.LABELS = list(R[0])
    settings.CONFIGURATIONS_LABELS = list(R[1])
    R = get_optimal_gains(settings.LABELS)        
    final = R[0]
    settings.OPTIMAL_CONFIG_LIST = list(R[1])
    settings.OPTIMAL_CONFIG_CODE = list(R[2])
    print("final gains;",final, "; size space;",len(settings.LABELS[0]))
    print()
        
    # This command does not depend on the parameters
    print("STUDY SPACE PROJECTION")
    project_space()
    print()
    print()
   
    print("MODEL PREDICTION")
   
    # manually select which input size we want for prediction
    SUBSET = "1"
    subset_input(SUBSET)           
    # example of running the ml model using feature 10 and 15
    # over input size 1 - small
    call_classify([10,15])
    
    # run the GA exploration
    # exlpore_GA(2500,25,0.9,0.1)
    
main()
