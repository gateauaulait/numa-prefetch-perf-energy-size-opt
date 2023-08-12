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

import settings

from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier
from statistics import mean
import statistics
import copy

fold_nb_code = []
fold_order = []

# Used to determine if a code should be to train or to validate
# Using fold distribution defined for time_med input 1
def select_train_validate_custom(train_code,validate_code,fold):
    id_use = "1time_med"
    
    nb_code = len(settings.GROUP_CODE[id_use])
    
    validate_code_custom = []
    train_code_custom = []
    
    if not fold_nb_code:        
        basic_count = math.trunc(nb_code/10)
        adjust_count = nb_code%10
        for x in range(10):
            if x >= adjust_count:
                fold_nb_code.append(basic_count)
            else:
                fold_nb_code.append(basic_count+1)
    
        for n,v in enumerate(fold_nb_code):
            for val in range(v):
                fold_order.append(n)    

    if not train_code_custom:
        random.seed(settings.SEED)
        list_codes = list(range(nb_code))
        random.shuffle(list_codes)
        
        for n,v in enumerate(fold_order):
            if v == fold:
                validate_code_custom.append(list_codes[n])
            else:
                train_code_custom.append(list_codes[n])

    for n,c in enumerate(settings.CODES):
        assert(c in settings.GROUP_CODE[id_use])
        ind = settings.GROUP_CODE[id_use].index(c)
        if ind in train_code_custom:
            train_code.append(n)
        else:
            validate_code.append(n)
        

# Used to determine if a code should be to train or to validate
# native fold construction
def select_train_validate(train_code,validate_code,fold):
    nb_code = len(settings.CODES)
    if not fold_nb_code:       
        basic_count = math.trunc(nb_code/10)
        adjust_count = nb_code%10
        for x in range(10):
            if x >= adjust_count:
                fold_nb_code.append(basic_count)
            else:
                fold_nb_code.append(basic_count+1)
    
        for n,v in enumerate(fold_nb_code):
            for val in range(v):
                fold_order.append(n)    

    if not train_code:
        random.seed(settings.SEED)
        list_codes = list(range(nb_code))
        random.shuffle(list_codes)
        
        for n,v in enumerate(fold_order):
            if v == fold:
                validate_code.append(list_codes[n])
            else:
                train_code.append(list_codes[n])

# initialize validation codes
# only use a subset of the features to make prediction
def init_model(train_code,validate_code,flitered_data,explore_list,fold):    
    select_train_validate_custom(train_code,validate_code,fold)
    #select_train_validate(train_code,validate_code,fold)

    # Subset only the individual that we want to explore
    for d in settings.FEATURES:
        l = []
        for f in explore_list:
            l.append(d[f])
        flitered_data.append(l)

    assert(len(flitered_data) == len(settings.FEATURES))

def fill_label_feature(feature,label,code,data,optimal_mapping_code):
    for c in code:
        feature.append(data[c])
        label.append(optimal_mapping_code[c])

def tree_training(X,Y):
    #clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)    
    return clf    

def validate_model(validate_code,predicted_label,validate_label,optimal_config_list,optimal_config_code):
    best = []
    measured = []
    
    for n,c in enumerate(validate_code):       
        codelet_id = c
        unique_mapping_id = optimal_config_code[c]
                
        best.append(settings.LABELS[c][optimal_config_list[optimal_config_code[c]]])        
        measured.append(settings.LABELS[c][optimal_config_list[predicted_label[n]]])
        
        # comment to not display results per code
        print(settings.CODES[c],settings.LABELS[c][optimal_config_list[predicted_label[n]]],settings.LABELS[c][optimal_config_list[optimal_config_code[c]]])
    # Remove comment to not show difference between folds
    print()
    return[scipy.stats.mstats.gmean(measured),scipy.stats.mstats.gmean(best),measured,best]

def classify(explore_list,fold):
    train_feature = []
    train_label = []
    train_code = []
    
    validate_feature = []
    validate_label = []
    validate_code = []

    flitered_data = []

    # filter data + set train validation codes
    init_model(train_code,validate_code,flitered_data,explore_list,fold) 

    # fill features / labels
    fill_label_feature(train_feature,train_label,train_code,flitered_data,settings.OPTIMAL_CONFIG_CODE)
    fill_label_feature(validate_feature,validate_label,validate_code,flitered_data,settings.OPTIMAL_CONFIG_CODE)    

    settings.CLF["tree"] = tree_training(train_feature,train_label)

    quality = validate_model(validate_code,copy.deepcopy(settings.CLF["tree"].predict(validate_feature)),validate_label,settings.OPTIMAL_CONFIG_LIST,settings.OPTIMAL_CONFIG_CODE)    
    return[quality[0],quality[1],quality[2],quality[3]]

def call_classify(explore_list):
    speedup_max = []
    speedup_predicted = []

    measured = []
    best = []
    
    for fold in range(10):
        ret = classify(explore_list,fold)

        speedup_predicted.append(ret[0])
        speedup_max.append(ret[1])
        
        for v in ret[2]:
            measured.append(v)

        for v in ret[3]:
            best.append(v)
    
    #print(explore_list,scipy.stats.mstats.gmean(speedup_predicted),scipy.stats.mstats.gmean(speedup_max))
    print(explore_list,statistics.mean(measured),statistics.mean(best))
    print()
 
    return statistics.mean(measured)
