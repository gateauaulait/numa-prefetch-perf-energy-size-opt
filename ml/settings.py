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

#import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.neighbors import kneighbors_graph
import sklearn

# Reproduce Isaac clustering

from scipy.cluster.hierarchy import centroid, fcluster
from scipy.spatial.distance import pdist
import scipy.stats.mstats

def init_data():
    global LABELS
    global FEATURES
    # Orders used in data to access elements
    global CONFIGURATIONS_LABELS    	# for labels
    global CONFIGURATIONS_FEATURES
    global CODES		# codes order in featres and labels    

    LABELS = []
    FEATURES = []        
    CONFIGURATIONS_LABELS = []
    CONFIGURATIONS_FEATURES = []
    CODES = []

    # Because data have different sources inputs/perf/energy
    # we group them according to their source
    global GROUP_LABELS
    global GROUP_FEATURES
    global GROUP_CODE
    GROUP_LABELS = {}
    GROUP_FEATURES = {}
    GROUP_CODE = {} 

    # We follow a similar structure for the quantify the instability
    global GROUP_LABELS_INSTABILITY
    global GROUP_CODE_INSTABILITY
    GROUP_LABELS_INSTABILITY = {}
    GROUP_CODE_INSTABILITY = {}

    global TARGET_TYPE
    global GOAL_INPUT
    global GOAL_TYPE
    TARGET_TYPE = ["time_med","energy"] 
    GOAL_INPUT = "1"
    GOAL_TYPE = "energy"
    GOAL_TYPE = "time_med"    
    
    global REGION_TO_SWITCH
    REGION_TO_SWITCH = ["__cere__3D_computeTempOMP_150","__cere__bfs__Z8BFSGraphiPPc_135","__cere__bfs__Z8BFSGraphiPPc_157","__cere__euler3d_cpu__Z12compute_fluxiPiPfS0_S0_S0_6float3S1_S1_S1__211","__cere__euler3d_cpu__Z9time_stepiiPfS_S_S__347","__cere__ex_particle_OPENMP_seq_particleFilter_371","__cere__ex_particle_OPENMP_seq_particleFilter_408","__cere__ex_particle_OPENMP_seq_particleFilter_486","__cere__hotspot_openmp__Z16single_iterationPfS_S_iifffff_69","__cere__kmeans_clustering_kmeans_clustering_183","__cere__lud_omp_lud_omp_123","__cere__main_main_253","__cere__main_main_295","__cere__needle__Z12nw_optimizedPiS_S_iii_116","__cere__needle__Z12nw_optimizedPiS_S_iii_176","__cere__pathfinder__Z3runiPPc_98","__cere__streamcluster_omp__Z5pgainlP6PointsdPliP17pthread_barrier_t_451","__cere__streamcluster_omp__Z5pgainlP6PointsdPliP17pthread_barrier_t_539","__cere____kernel_kernel_cpu_kernel_cpu_112"]

    global TRAINING_VALIDATION    
    global ABSOLUTE_PATH

    TRAINING_VALIDATION = []
    ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))               
    
    global SEED
    global CLF 
    global THRESHOLD_TRAINING
    global NB_LABELS    
    global THREADS
    global INPUTS
    global THRESHOLD_INSTABILITY

    # the seed impacts the quality of the predictions
    # this is something to further investigate / quanitfy
    SEED = 2
    np.random.seed(SEED)    
    CLF = {}     # training model used to make cmp
    THRESHOLD_TRAINING = 0.95
    NB_LABELS = 10
    THREADS = [16,32,64]    
    INPUTS = ["1","2"]
    THRESHOLD_INSTABILITY = 0.25

    global OPTIMAL_CONFIG_LIST # list of best config to be used for labels
    global OPTIMAL_CONFIG_CODE # list of best config per code relative to the OPTIMAL_CONFIG_LIST 

    OPTIMAL_CONFIG_LIST = []
    OPTIMAL_CONFIG_CODE = []