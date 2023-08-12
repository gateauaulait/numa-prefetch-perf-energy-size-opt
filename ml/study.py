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

import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.neighbors import kneighbors_graph
import sklearn

from scipy.cluster.hierarchy import centroid, fcluster
from scipy.spatial.distance import pdist
import scipy.stats.mstats

from statistics import geometric_mean
import statistics
from scipy.stats import gmean
import copy

import settings

import seaborn as sns; sns.set_theme(style='white')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator

from sklearn.decomposition import PCA

# return 
# settings.OPTIMAL_CONFIG_LIST => list of best conf for labels
# OPTIMAL_CONFIG_CODE => list of labels for the ML
def get_optimal_gains(labels):
    local_OPTIMAL_CONFIG_LIST = []
    local_OPTIMAL_CONFIG_CODE = []
    best_conf = -1
    best_conf_last = -2
    best_conf_index = -1
    best_conf_index_save = []
    
    INDEX_USE = range(len(labels[0]))
    
    nb_conf = 0
    while(best_conf_last != best_conf and nb_conf < settings.NB_LABELS):
        best_conf_last = best_conf
        best_conf = -1 
        for n,x in enumerate(INDEX_USE):
            val = []
            for T in labels:
                local_max = T[n]
                for j in best_conf_index_save:
                    if local_max < T[j]:
                        local_max = T[j]
                val.append(local_max)
            if best_conf < statistics.mean(val):
                     best_conf = statistics.mean(val)
                     best_conf_index = n    
   
        nb_conf = nb_conf +1
        best_conf_index_save.append(best_conf_index)        
        #print(nb_conf,';',best_conf_index_save,";",best_conf)  

    local_OPTIMAL_CONFIG_LIST = copy.deepcopy(best_conf_index_save)             
    
    for n,c in enumerate(labels):
        local_max = -1
        local_max_index = -1
        for v in local_OPTIMAL_CONFIG_LIST:
            if labels[n][v] >  local_max:
                local_max = labels[n][v]
                local_max_index = v

        local_OPTIMAL_CONFIG_CODE.append(local_OPTIMAL_CONFIG_LIST.index(local_max_index))
    
    return [best_conf,local_OPTIMAL_CONFIG_LIST,local_OPTIMAL_CONFIG_CODE]
        
# only consider features from a given scale/input
def subset_input(target_input):
    assert(len(settings.CONFIGURATIONS_FEATURES) == len(settings.FEATURES[0]))
    feature_head = []
    for m,v in enumerate(settings.FEATURES):                        
        feature_value = []
        for n,f in enumerate(settings.CONFIGURATIONS_FEATURES):
            if m == 0 and (settings.CONFIGURATIONS_FEATURES[n][0] == target_input):
                feature_head.append(settings.CONFIGURATIONS_FEATURES[n])
            if settings.CONFIGURATIONS_FEATURES[n][0] == target_input:
                feature_value.append(settings.FEATURES[m][n])                
    
        settings.FEATURES[m] = copy.deepcopy(feature_value)
        

    settings.CONFIGURATIONS_FEATURES = copy.deepcopy(feature_head)            
    assert(len(settings.CONFIGURATIONS_FEATURES) == len(settings.FEATURES[0]))        

class Configuration:
    def __init__(self):
        self.th = 0
        self.node = 0
        self.prefetch = 0
        self.hyperthread = 0
        self.thread_mapping = 0
        self.data= 0

    def update(self,conf):
        l = conf.split("_")
        self.th = int(l[1])
        assert(l[2] == l[3])
        self.node = int(l[3])    
        self.prefetch = int(l[6])
        self.hyperthread = int(l[5])    
        
        self.data = l[7].replace("\n","")
        self.thread_mapping = l[4].replace("\n","")
        
    def print_conf(self):
        print("threads:", self.th)
        print("node:", self.node)
        print("prefetch:", self.prefetch)
        print("hyperthread:",self.hyperthread)
        print("thread_mapping",self.thread_mapping)
        print("data_mapping",self.data)

    def filter_threads(self,element):
        if element == "":
            return True            
        if self.th != element:
            return False
        else: 
            return True

    def filter_node(self,element):
        if element == "":
            return True  
        if self.node != element:
            return False
        else: 
            return True

    def filter_prefetch(self,element):
        if element == "":
            return True  
        if self.prefetch != element:
            return False
        else: 
            return True

    def filter_hyperthread(self,element):
        if element == "":
            return True  
        if self.hyperthread != element:
            return False
        else: 
            return True

    def filter_thread_mapping(self,element):
        if element == "":
            return True  
        if self.thread_mapping != element:
            return False
        else: 
            return True

    def filter_data(self,element):
        if element == "":
            return True  
        if self.data != element:
            return False
        else: 
            return True

# We only need to consider data from the target group
# we quantify the gains across different search spaces
# we update CONFIGURATIONS_LABELS and LABELS
def subset_labels(labels_ori,configuration_labels,sub_th,sub_node,sub_pref,sub_hyper,sub_th_map,sub_data):
    labels = copy.deepcopy(labels_ori)
    assert(len(configuration_labels) == len(labels[0]))
        
    labels_to_keep = []
    
    for n,lab in enumerate(configuration_labels):
        x = Configuration()
        x.update(lab)
        # default configuration
        if(x.filter_threads(sub_th) and x.filter_node(sub_node) and x.filter_prefetch(sub_pref) and x.filter_hyperthread(sub_hyper) and x.filter_thread_mapping(sub_th_map) and x.filter_data(sub_data)):
        #if(x.filter_threads(32) and x.filter_node(2) and x.filter_prefetch(0) and x.filter_hyperthread(0) and x.filter_thread_mapping('scatter') and x.filter_data("ori")):
            labels_to_keep.append(n)
    
    TMP_CONFIGURATIONS_LABELS = []    
    for n in labels_to_keep:
        TMP_CONFIGURATIONS_LABELS.append(configuration_labels[n])
    local_CONFIGURATIONS_LABELS = copy.deepcopy(TMP_CONFIGURATIONS_LABELS)    
                
    for m,c in enumerate(labels):
        TMP_LABELS = []
        for n in labels_to_keep:
            TMP_LABELS.append(labels[m][n])
        labels[m] = copy.deepcopy(TMP_LABELS)

    assert(len(local_CONFIGURATIONS_LABELS) == len(labels[0]))  
    return([labels,local_CONFIGURATIONS_LABELS])

    
# returns the gains of using a set of configurations
# configurations are independently selected for the code
# can also evaluate the performance per region
# to be used to estimate emerging behaviors
def calculate_gains_optimal(L,codes,labels):
    assert(len(codes) == len(labels))

    final = []
    for n,c in enumerate(codes):
        current = []
        for conf in L:
            current.append(labels[n][conf])
        print(c,";",max(current))
        final.append(max(current))
        
    print("gains:",statistics.mean(final),gmean(final))   
    print()

# returns the gains of using a set of configurations
# configurations fixed for each code
# can also evaluate the performance per region
# to be used to estimate emerging behaviors
def calculate_gains(L,codes,labels):
    assert(len(L) == len(codes))
    assert(len(codes) == len(labels))
    
    final = []
    for n,c in enumerate(codes):        
        final.append(labels[n][L[n]])
        print(c,";",labels[n][L[n]])
        
    print("gains:",statistics.mean(final),gmean(final))   
    print()


# input 1: codes A
# input 2: codes B
# Go over the codes 1 and return a sub group list of codes common to both A and B
def get_common_codes(codesA,codesB):
    common = []
    for n,c in enumerate(codesA):
        if c in codesB:
            common.append(c)
    return common
        
# input 1: group of codes from get common codes
# input 2 and 3: list of codes along labels 
# create the resulting labels following the order of input 1
def create_labels(codes_order,codes,labels):
    labels_return = []
    for c in codes_order:
        index = codes.index(c)
        labels_return.append(copy.deepcopy(labels[index]))
    return labels_return
    
# input 1: list of best conf in OPTIMAL_CONFIG_CODE 
# input 2: list of best conf in OPTIMAL_CONFIG_LIST
# input 3: list of codes
# output: list of best conf per code according to original labels
def translate_confs_from_optimal_gains(opt_config_code, opt_config_list):
    labels_use = []    
    for conf in opt_config_code:
        labels_use.append(opt_config_list[conf])   
    # print(labels_use)   
    return labels_use

def generate_labels_to_use(labels):
    R = get_optimal_gains(labels)    
    conf_list = copy.deepcopy(R[1])
    conf_code = copy.deepcopy(R[2])
    labels_use = translate_confs_from_optimal_gains(conf_code,conf_list)    
    return labels_use

# Evaluate the sub-spaces gains
def subset_space(common_labels,I,T):         

    ####### 1 Dimension
    # sub_th,sub_node,sub_pref,sub_hyper,sub_th_map,sub_data
    # subset_labels(32,2,0,0,"scatter","ori")  
    R = subset_labels(common_labels[I+T],settings.CONFIGURATIONS_LABELS,32,2,0,0,"scatter","ori")
    size = len(R[1])
    R = get_optimal_gains(R[0])        
    final = R[0]
    print("size:",size,"GAINS OF EXPERIMENT",I+T,"IS",final,"by considering default")

    R = subset_labels(common_labels[I+T],settings.CONFIGURATIONS_LABELS,"",2,0,0,"scatter","ori")
    size = len(R[1])
    R = get_optimal_gains(R[0])        
    final = R[0]
    print("size:",size,"GAINS OF EXPERIMENT",I+T,"IS",final,"by considering parallelism")

    R = subset_labels(common_labels[I+T],settings.CONFIGURATIONS_LABELS,32,"",0,0,"scatter","ori")
    size = len(R[1])
    R = get_optimal_gains(R[0])        
    final = R[0]
    print("size:",size,"GAINS OF EXPERIMENT",I+T,"IS",final,"by considering # node")

    R = subset_labels(common_labels[I+T],settings.CONFIGURATIONS_LABELS,32,2,"",0,"scatter","ori")
    size = len(R[1])
    R = get_optimal_gains(R[0])        
    final = R[0]
    print("size:",size,"GAINS OF EXPERIMENT",I+T,"IS",final,"by considering prefetchers")

    R = subset_labels(common_labels[I+T],settings.CONFIGURATIONS_LABELS,32,2,0,"","scatter","ori")
    size = len(R[1])
    R = get_optimal_gains(R[0])        
    final = R[0]
    print("size:",size,"GAINS OF EXPERIMENT",I+T,"IS",final,"by considering hyper threading")

    R = subset_labels(common_labels[I+T],settings.CONFIGURATIONS_LABELS,32,2,0,0,"","ori")
    size = len(R[1])
    R = get_optimal_gains(R[0])        
    final = R[0]
    print("size:",size,"GAINS OF EXPERIMENT",I+T,"IS",final,"by considering thread mapping")

    R = subset_labels(common_labels[I+T],settings.CONFIGURATIONS_LABELS,32,2,0,0,"scatter","")
    size = len(R[1])
    R = get_optimal_gains(R[0])        
    final = R[0]
    print("size:",size,"GAINS OF EXPERIMENT",I+T,"IS",final,"by considering data")

    R = subset_labels(common_labels[I+T],settings.CONFIGURATIONS_LABELS,32,"",0,"","","ori")
    size = len(R[1])
    R = get_optimal_gains(R[0])        
    final = R[0]
    print("size:",size,"GAINS OF EXPERIMENT",I+T,"IS",final,"by considering threads")


    ####### 2 Dimensions

    R = subset_labels(common_labels[I+T],settings.CONFIGURATIONS_LABELS,32,"",0,"","","")
    size = len(R[1])
    R = get_optimal_gains(R[0])        
    final = R[0]
    print("size:",size,"GAINS OF EXPERIMENT",I+T,"IS",final,"by considering data + thread")

    R = subset_labels(common_labels[I+T],settings.CONFIGURATIONS_LABELS,32,2,"",0,"scatter","")
    size = len(R[1])
    R = get_optimal_gains(R[0])        
    final = R[0]
    print("size:",size,"GAINS OF EXPERIMENT",I+T,"IS",final,"by considering data + prefetch")

    R = subset_labels(common_labels[I+T],settings.CONFIGURATIONS_LABELS,"",2,0,0,"scatter","")
    size = len(R[1])
    R = get_optimal_gains(R[0])        
    final = R[0]
    print("size:",size,"GAINS OF EXPERIMENT",I+T,"IS",final,"by considering data + para")

    R = subset_labels(common_labels[I+T],settings.CONFIGURATIONS_LABELS,32,"","","","","ori")
    size = len(R[1])
    R = get_optimal_gains(R[0])        
    final = R[0]
    print("size:",size,"GAINS OF EXPERIMENT",I+T,"IS",final,"by considering thread + prefetch")

    R = subset_labels(common_labels[I+T],settings.CONFIGURATIONS_LABELS,"","",0,"","","ori")
    size = len(R[1])
    R = get_optimal_gains(R[0])        
    final = R[0]
    print("size:",size,"GAINS OF EXPERIMENT",I+T,"IS",final,"by considering thread + para")

    R = subset_labels(common_labels[I+T],settings.CONFIGURATIONS_LABELS,"",2,"",0,"scatter","ori")
    size = len(R[1])
    R = get_optimal_gains(R[0])        
    final = R[0]
    print("size:",size,"GAINS OF EXPERIMENT",I+T,"IS",final,"by considering pref + para")

        
    ####### 3 Dimensions

    R = subset_labels(common_labels[I+T],settings.CONFIGURATIONS_LABELS,"","",0,"","","")
    size = len(R[1])
    R = get_optimal_gains(R[0])        
    final = R[0]
    print("size:",size,"GAINS OF EXPERIMENT",I+T,"IS",final,"by considering para + thread + data")

    R = subset_labels(common_labels[I+T],settings.CONFIGURATIONS_LABELS,"","","","","","ori")
    size = len(R[1])
    R = get_optimal_gains(R[0])        
    final = R[0]
    print("size:",size,"GAINS OF EXPERIMENT",I+T,"IS",final,"by considering para + thread + pref") 
    
    R = subset_labels(common_labels[I+T],settings.CONFIGURATIONS_LABELS,32,"","","","","")
    size = len(R[1])
    R = get_optimal_gains(R[0])        
    final = R[0]
    print("size:",size,"GAINS OF EXPERIMENT",I+T,"IS",final,"by considering th + pref + data")       

    ####### all 4  Dimensions

    # sub_th,sub_node,sub_pref,sub_hyper,sub_th_map,sub_data
    # subset_labels(32,2,0,0,"scatter","ori")  
    R = subset_labels(common_labels[I+T],settings.CONFIGURATIONS_LABELS,"","","","","","")
    size = len(R[1])
    R = get_optimal_gains(R[0])        
    final = R[0]
    print("size:",size,"GAINS OF EXPERIMENT",I+T,"IS",final,"by considering all")

def create_code_size(common_labels,common_codes):
    # 2d list 
    # [code,mem size] [code,mem size] [code,mem size] [code,mem size] 
    # has codes from both sizes
    code_size = []
    
    assert(len(settings.FEATURES) == len(settings.CODES))
    assert(len(settings.FEATURES[0]) == len(settings.CONFIGURATIONS_FEATURES))
    
    for c in common_codes:
        id_code = settings.CODES.index(c)
        id_size1 = settings.CONFIGURATIONS_FEATURES.index("1mem-footprint-16")
        id_size2 = settings.CONFIGURATIONS_FEATURES.index("2mem-footprint-16")
       
        small_size = 0
        big_size = 0
        if(settings.FEATURES[id_code][id_size1] < settings.FEATURES[id_code][id_size2]):
            small_size = settings.FEATURES[id_code][id_size1]
            big_size = settings.FEATURES[id_code][id_size2]
        else:
            small_size = settings.FEATURES[id_code][id_size2]
            big_size = settings.FEATURES[id_code][id_size1]  
            
        l = []
        l.append("1"+c)
        l.append(small_size)
        code_size.append(l)
        l = []
        l.append("2"+c)
        l.append(big_size)
        code_size.append(l)
    
    L = copy.deepcopy(sorted(code_size,key=lambda x: (x[1])))
    R = []
    for i in L:
        R.append(i[0])

    return R

def collect_perf_energy(conf_list,code_size,common_labels,common_codes,conf_type):
    # 2d list 
    # list of codes ordered according to code_size
    # for each code list of perf/energy ordered according to conf_list
    # do it in for the two lists of confs, one per size
    L = []
    
    assert(len(common_labels["1energy"][0]) == len(settings.CONFIGURATIONS_LABELS))
    assert(len(common_labels["2time_med"][0]) == len(settings.CONFIGURATIONS_LABELS))
    
    for c in code_size:
        size = int(c[0])
        region = c[1:]
        region_index = common_codes.index(region)
        l = []
        for f in conf_list:
            l.append(common_labels[str(size)+conf_type][region_index][f])
        L.append(l)    
    
    return L

# main function to understand space projection
def project_space():   
    labels = copy.deepcopy(settings.LABELS)
    codes = copy.deepcopy(settings.CODES)
    
    labels_use = copy.deepcopy(generate_labels_to_use(labels))
    calculate_gains(labels_use,codes,labels)    

    # create list of common codes across all groups
    common_codes = get_common_codes(settings.GROUP_CODE["1energy"],settings.GROUP_CODE["2energy"])
    common_codes = get_common_codes(settings.GROUP_CODE["1time_med"],common_codes)
    common_codes = get_common_codes(settings.GROUP_CODE["2time_med"],common_codes)

    # time issue size 2 - 0 values for the execution / cross size study not possible as a result 
    common_codes.remove("__cere__3D_computeTempOMP_150")

    # create the labels for each group
    common_labels = {}
    for I in settings.INPUTS:
        for T in settings.TARGET_TYPE:            
            common_labels[I+T] = create_labels(common_codes, settings.GROUP_CODE[I+T],settings.GROUP_LABELS[I+T])
            assert(len(common_codes) == len(common_labels[I+T]))

    # evaluate gains across sub spaces
    print("Evaluate gains across sub-spaces")
    for I in settings.INPUTS:
        for T in settings.TARGET_TYPE:             
            labels_use = copy.deepcopy(generate_labels_to_use(common_labels[I+T]))           
            print("We consider scenario:",I,T)
            calculate_gains(labels_use,common_codes,common_labels[I+T])            
            subset_space(common_labels,I,T)
            print()
    print()
    print()
    print()
    
    # project all spaces
    for I1 in settings.INPUTS:
        for T1 in settings.TARGET_TYPE:
            for I2 in settings.INPUTS:
                for T2 in settings.TARGET_TYPE:
                    # taking I1 T1 labels and applying to I2 T2 labels
                    labels_use = copy.deepcopy(generate_labels_to_use(common_labels[I1+T1]))
                    print("PROJECTING ",I1+T1,"OVER THE SPACE",I2+T2)
                    calculate_gains(labels_use,common_codes,common_labels[I2+T2])
    

