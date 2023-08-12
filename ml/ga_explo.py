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

from scipy.cluster.hierarchy import centroid, fcluster
from scipy.spatial.distance import pdist
import scipy.stats.mstats
from classify import call_classify


import settings

# indivudual for GA exploration 
def create_individual(data):
    l = list([random.randint(0, len(settings.FEATURES[0])-1) for _ in range(2)])
    return l

def training_GA(explore_list,data):    
    r = call_classify(explore_list)
    #print(r,explore_list)   
    return r    


def exlpore_GA(ps,gen,cross,mut):
    global XXX_training_type
    global XXX_many_labels
        
    ga = pyeasyga.GeneticAlgorithm([],
                               population_size=ps,
                               generations=gen,
                               crossover_probability=cross,
                               mutation_probability=mut,
                               #elitism=True,
                               maximise_fitness=True)
                               

    ga.create_individual = create_individual   
    ga.fitness_function = training_GA
    ga.run()
    print (ga.best_individual())    

