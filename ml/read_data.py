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
from scipy.stats import gmean
import statistics
import copy

import settings

# function used to execute bash commands
# returns as text the output of the command
def exec_cmd(cmd):
    #print (cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,shell=True)
    result = p.communicate()[0]
    p.wait() 
    return result

# check if s is a number
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False


# given a file name collect: 
# labels / label-conf / code
# this function collects information for both values and stability
def collect_label_conf_code(file_name,list_data,list_head,list_code):
    f = open(file_name, "r")
    
    for n,i in enumerate(f):
        l = i.split(";")
        if n == 0:
            if len (list_head) == 0:
                for m,j in enumerate(l):
                    if m >0 :
                        list_head.append(j)
        else:
            values_labels = []
            for m,j in enumerate(l):
                if m == 0:
                    list_code.append(j) 
                else:
                    if is_number(j):
                        values_labels.append(float(j))
                    else:
                        values_labels.append(-1)
           
            if (values_labels[len(values_labels)-1] == -1):
                values_labels.pop()
                
            assert(len(values_labels) == len(list_head))
            list_data.append(values_labels)

    # because of the support of multiple inputs
    # we cannot assert a fixed number of codes
    #assert(len(list_code) == len(list_data))
    
    f.close() 

# Collect labels for the model
# either time or energy
# Need to generate content for:
# CODES => the order of the benchmarks
# CONFIGURATUION_LABELS: the order of the configurations for the labels
# LABELS: fill with the actual values
def collect_label():	
    # Collect values
    os.chdir(settings.ABSOLUTE_PATH+"/../data-executions-time-energy/values")
    
    # We iterate over all the sources of data: inputs and type (perf/energy) 
    for ts in settings.TARGET_TYPE:
        for size in settings.INPUTS:
            file_name = "conf"+size+"_"+ts+".csv"	        
            list_data = []
            list_head = []
            list_code = []
            
            # for each type we collect the raw values
            collect_label_conf_code(file_name,list_data,list_head,list_code)
            
            # we initialize group structure to contain all the data per type of source
            settings.GROUP_LABELS[size+ts] = copy.deepcopy(list_data)
            
            # all the source share the same label order: we can store it once 
            if not settings.CONFIGURATIONS_LABELS:
                settings.CONFIGURATIONS_LABELS = copy.deepcopy(list_head) 
            assert(settings.CONFIGURATIONS_LABELS == list_head)
            settings.GROUP_CODE[size+ts] = copy.deepcopy(list_code)       
    
    os.chdir(settings.ABSOLUTE_PATH)

    # Repeat work but on instability this time
    os.chdir(settings.ABSOLUTE_PATH+"/../data-executions-time-energy/stability")
    for ts in settings.TARGET_TYPE:
        for size in settings.INPUTS:
            file_name = "conf"+size+"_"+ts+".csv"	        
            list_data = []
            list_head = []
            list_code = []
            collect_label_conf_code(file_name,list_data,list_head,list_code)
            
            settings.GROUP_LABELS_INSTABILITY[size+ts] = copy.deepcopy(list_data)
            settings.GROUP_CODE_INSTABILITY[size+ts] = copy.deepcopy(list_code)
            
            assert(set(settings.GROUP_CODE_INSTABILITY[size+ts]) == set(settings.GROUP_CODE[size+ts]))
            assert(settings.CONFIGURATIONS_LABELS == copy.deepcopy(list_head))        
    os.chdir(settings.ABSOLUTE_PATH)

# Collect features for the model
def collect_feature_conf(file_name,codes):
    f = open(file_name, "r")
    
    features = []
    
    # temporary to the function data to collect features
    # use dictionary because we do not have info about the order of the regions within the feature csv
    # data layout:
    # DATA[region][input size]
    DATA = {}  

    for n,i in enumerate(f):
        l = i.split(",")
        conf_features = []
        if(n == 0):
            if len(settings.CONFIGURATIONS_FEATURES) == 0:
                for inp in settings.INPUTS:
                    for th in settings.THREADS:
                        for c in l[3:]:
                            settings.CONFIGURATIONS_FEATURES.append(inp + "-" + str(th) + "-"+c.replace("\n",""))
        else:                        
            values_features = []
            if l[0] not in DATA.keys(): # region
                DATA[l[0]] = {}
            if l[1] not in DATA[l[0]].keys(): # input
                DATA[l[0]][l[1]] = {}
            if l[2] not in DATA[l[0]][l[1]].keys(): # threads
                DATA[l[0]][l[1]][l[2]] = []
                for v in l[3:]:
                    values_features.append(float(v))
                DATA[l[0]][l[1]][l[2]] = copy.deepcopy(values_features)
        
    for c in codes:
        usefull = True
        assert (c in DATA.keys())
        values_features = []
        if c in DATA.keys():
            inp_traverse = []            
            if c not in settings.REGION_TO_SWITCH:
                inp_traverse = copy.deepcopy(settings.INPUTS)           
            else:
                inp_traverse = ["2","1"]
            
            for inp in inp_traverse:
                if ("conf"+inp not in DATA[c].keys()):
                    #print("MISSING FEATURES INPUTS FOR REGION: ",c,DATA[c].keys(),"conf"+inp ,"is missing" )
                    usefull = False
                else:
                    for th in settings.THREADS:
                        assert(str(th) in DATA[c]["conf"+inp])
                        for v in DATA[c]["conf"+inp][str(th)]:
                            values_features.append(float(v))
        if (usefull):
            assert(len(values_features) == len(settings.CONFIGURATIONS_FEATURES))
            features.append(values_features)
        else:
            features.append([])
    assert(len(codes) == len(features))        
          
    f.close() 
    
    return features
    
# Collect feature and configurations features
def collect_feature(codes):
    os.chdir(settings.ABSOLUTE_PATH+"/../communication-matrix/")
    
    file_name = "statistics-comm-page.csv"
    R = collect_feature_conf(file_name,codes)    
    
    os.chdir(settings.ABSOLUTE_PATH)
    
    return R

# remove codes that have either 0 in their vectors or have missing features
def remove_empty_codes(codes,labels,features):
    TMP_CODE = []
    TMP_LABELS = []    
    TMP_FEATURES = []
    
    for n,c in enumerate(codes):  
        if len(features[n]) == 0 or 0 in labels[n]:
            pass
            #print("filtered away on target:",c,len(features[n]),len(labels[n]))
        else:
            TMP_CODE.append(c)
            TMP_LABELS.append(labels[n])
            TMP_FEATURES.append(copy.deepcopy(features[n]))
      
    
    codes = copy.deepcopy(TMP_CODE)
    labels = copy.deepcopy(TMP_LABELS)
    features = copy.deepcopy(TMP_FEATURES)
    return [codes,labels,features]
            

# function has already been executed
# it is not called because it takes a lot of time to compute
# directly returned its result in the code: see settings.py REGION_TO_SWITCH
# __cere__is_rank_486 has a unique behavior, only requiring to be changed with CLASS B
def get_codes_to_switch():
    directory = settings.ABSOLUTE_PATH+"/../communication-matrix/input-ml-explo-trace"
    directory_contents = os.listdir(directory)
    for d in directory_contents:
        os.chdir(d)
        if os.path.isdir("conf1") and os.path.isdir("conf2") and d == "__cere__is_rank_486":
            s1 = int(str(exec_cmd("wc -l conf1/trace/numalize"+d+"_0_16.csv")).split(" ")[0].replace("b'",""))
            s2 = int(str(exec_cmd("wc -l conf2/trace/numalize"+d+"_0_16.csv")).split(" ")[0].replace("b'",""))
            if s2 < s1:
                pass
                #print("to switch:",d,s1,s2)
            #print("values:",d,s1,s2)
        os.chdir("../")

# only collect relevant codes for the target
# switch around codes based on their memory footprint
def adjust_codes(group,group_code):
    directory = settings.ABSOLUTE_PATH+"/../communication-matrix/input-ml-explo-trace"
    os.chdir(directory)

    # get_codes_to_switch()
    # print(settings.REGION_TO_SWITCH)

    DATA_LABELS_TMP = {}

    for I in settings.INPUTS:
        for T in settings.TARGET_TYPE:
            DATA_LABELS_TMP[I+T] = {}
            for n,c in enumerate(group_code[I+T]):
                if c in settings.REGION_TO_SWITCH:
                    DATA_LABELS_TMP[I+T][c] = copy.deepcopy(group[I+T][n])
    
    # switch entries for codes marked
    for T in settings.TARGET_TYPE:        
        for n,c in enumerate(group_code["1"+T]):
            if c in settings.REGION_TO_SWITCH:
                group["1"+T][n] = copy.deepcopy(DATA_LABELS_TMP["2"+T][c].copy())
                #print("changed code order from 1 to 2:",c,T,n)

    for T in settings.TARGET_TYPE:        
        for n,c in enumerate(group_code["2"+T]):
            if c in settings.REGION_TO_SWITCH:
                group["2"+T][n] = copy.deepcopy(DATA_LABELS_TMP["1"+T][c].copy())
                #print("changed code order from 2 to 1:",c,T,n)
    
    os.chdir(settings.ABSOLUTE_PATH)
    
    return [group, group_code]

# get codes memory footprint as a feature
def get_mem_footprint(code,th,inp):
    directory = settings.ABSOLUTE_PATH+"/../communication-matrix/input-ml-explo-trace/"
    #print("wc -l "+directory+"/"+code+"/conf"+inp+"/trace/numalize"+code+"_0_"+th+".csv")
    s = int(str(exec_cmd("wc -l "+directory+"/"+code+"/conf"+inp+"/trace/numalize"+code+"_0_"+th+".csv")).split(" ")[0].replace("b'",""))
    return s

# optional function, used to add features including memory footprint and reaction based counters
def augment_goal_features():   
    print("Augment data features -- this may take some time")
    
    # mem footprint
    NEW_features = []
    for n,c in enumerate(settings.CODES):        
        inp_traverse = []  
        if c not in settings.REGION_TO_SWITCH:
            inp_traverse = copy.deepcopy(settings.INPUTS)           
        else:
            inp_traverse = ["2","1"]
        
        if n == 0:
            for inp in inp_traverse:
                for th in settings.THREADS:
                    NEW_features.append(inp+"mem-footprint-"+str(th))        
            for f in NEW_features:
                settings.CONFIGURATIONS_FEATURES.append(f)           
        
        value_features = []                                 
        for inp in inp_traverse:
            for th in settings.THREADS:
                value_features.append(get_mem_footprint(c,str(th),inp))
        # print(c,len(value_features),value_features)
        assert(len(NEW_features) == len(value_features))
        for f in value_features:
            settings.FEATURES[n].append(f)

    assert(len(settings.FEATURES[0]) == len(settings.CONFIGURATIONS_FEATURES))
    
    # execution time + energy as counters    
    NEW_features = []    
    TMP_CONFIGURATIONS_LABELS = copy.deepcopy(settings.CONFIGURATIONS_LABELS)
    NEW_features = []
    for I in settings.INPUTS:
        for T in settings.TARGET_TYPE:
            for f in TMP_CONFIGURATIONS_LABELS:
                nf = I+T+f
                NEW_features.append(nf)
    for f in NEW_features:
        settings.CONFIGURATIONS_FEATURES.append(f)
    
    for n,c in enumerate(settings.CODES):
        value_features = []
        for I in settings.INPUTS:
            for T in settings.TARGET_TYPE:
                if c in settings.GROUP_CODE[I+T]:
                    for f in  settings.GROUP_LABELS[I+T][settings.GROUP_CODE[I+T].index(c)]:                       
                        value_features.append(f)
                else:
                    for f in range(len(TMP_CONFIGURATIONS_LABELS)):
                        value_features.append(-1)
        assert(len(NEW_features) == len(value_features))
        for f in value_features:
            settings.FEATURES[n].append(f)
            
    assert(len(settings.FEATURES[0]) == len(settings.CONFIGURATIONS_FEATURES))

# Normalize the labels 
# Can select the default configuration           
def normalize_labels(labels):    
    #index = settings.CONFIGURATIONS_LABELS.index("-O3_32_2_2_scatter_0_0_ori")
    index = settings.CONFIGURATIONS_LABELS.index("-O3_32_2_2_compact_0_0_loc")
    for n,v in enumerate(labels):
        norm_val = labels[n][index]
        for m,vv in enumerate(labels[n]):
            labels[n][m] = norm_val / labels[n][m]
    return labels

# Normalize the features
def normalize_feature_numalize():
    max_feature_val = []
    for i in settings.CONFIGURATIONS_FEATURES:
        max_feature_val.append(0)
    
    for n,v in enumerate(settings.FEATURES):
        for m,vv in enumerate(settings.FEATURES[n]):
            if settings.FEATURES[n][m] > max_feature_val[m]:
                max_feature_val[m] = settings.FEATURES[n][m]

    for n,v in enumerate(settings.FEATURES):
        for m,vv in enumerate(settings.FEATURES[n]):
            if max_feature_val[m] != 0:
                settings.FEATURES[n][m] = settings.FEATURES[n][m] / max_feature_val[m]
            else:
                settings.FEATURES[n][m] = 0

# codes filtered away due to instability
def remove_unstable_code(codes,labels,features,instability_code,instability_label):
    TMP_CODE = []
    TMP_LABELS = []    
    TMP_FEATURES = []
    
    # list of codes that we need to filter away given a threshold
    codes_to_filter = []
    
    for n,c in enumerate(instability_label):  
        if statistics.mean(instability_label[n]) > settings.THRESHOLD_INSTABILITY:
             codes_to_filter.append(instability_code[n])
                
    for n,c in enumerate(codes):  
        if c in codes_to_filter:
            pass
            #print("filtered away DUE TO INSTABILITY:",c,statistics.mean(instability_label[instability_code.index(c)]))
        else:
            TMP_CODE.append(c)
            TMP_LABELS.append(labels[n])
            TMP_FEATURES.append(copy.deepcopy(features[n]))    
    
    codes = copy.deepcopy(TMP_CODE)
    labels = copy.deepcopy(TMP_LABELS)
    features = copy.deepcopy(TMP_FEATURES)
    return [codes,labels,features]
    
                 
def read_info():
    collect_label()

    # R[0] => groups 
    # R[1] => codes
    R = adjust_codes(settings.GROUP_LABELS,settings.GROUP_CODE)
    settings.LABELS = copy.deepcopy(R[0][settings.GOAL_INPUT+settings.GOAL_TYPE]) 
    settings.CODES = copy.deepcopy(R[1][settings.GOAL_INPUT+settings.GOAL_TYPE])    
    for I in settings.INPUTS:
        for T in settings.TARGET_TYPE:
            settings.GROUP_LABELS[I+T] = copy.deepcopy(R[0][I+T])
            settings.GROUP_CODE[I+T] = copy.deepcopy(R[1][I+T])
            
    R = adjust_codes(settings.GROUP_LABELS_INSTABILITY,settings.GROUP_CODE_INSTABILITY)
    for I in settings.INPUTS:
        for T in settings.TARGET_TYPE:
            settings.GROUP_LABELS_INSTABILITY[I+T] = copy.deepcopy(R[0][I+T])
            settings.GROUP_CODE_INSTABILITY[I+T] = copy.deepcopy(R[1][I+T])
    
    for I in settings.INPUTS:
        for T in settings.TARGET_TYPE:
            settings.GROUP_FEATURES[I+T] = copy.deepcopy(collect_feature(settings.GROUP_CODE[I+T]))
    settings.FEATURES = copy.deepcopy(collect_feature(settings.CODES))
    
    for I in settings.INPUTS:
        for T in settings.TARGET_TYPE:
            R = remove_empty_codes(settings.GROUP_CODE[I+T],settings.GROUP_LABELS[I+T],settings.GROUP_FEATURES[I+T])
            settings.GROUP_CODE[I+T] = copy.deepcopy(R[0])
            settings.GROUP_LABELS[I+T] = copy.deepcopy(R[1])
            settings.GROUP_FEATURES[I+T]  = copy.deepcopy(R[2])
    R = remove_empty_codes(settings.CODES,settings.LABELS,settings.FEATURES)
    settings.CODES = copy.deepcopy(R[0])
    settings.LABELS = copy.deepcopy(R[1])
    settings.FEATURES = copy.deepcopy(R[2])     

    for I in settings.INPUTS:
        for T in settings.TARGET_TYPE:
            R = remove_unstable_code(settings.GROUP_CODE[I+T],settings.GROUP_LABELS[I+T],settings.GROUP_FEATURES[I+T],settings.GROUP_CODE_INSTABILITY[I+T],settings.GROUP_LABELS_INSTABILITY[I+T])
            settings.GROUP_CODE[I+T] = copy.deepcopy(R[0])
            settings.GROUP_LABELS[I+T] = copy.deepcopy(R[1])
            settings.GROUP_FEATURES[I+T]  = copy.deepcopy(R[2])    

    R = remove_unstable_code(settings.CODES,settings.LABELS,settings.FEATURES,settings.GROUP_CODE_INSTABILITY[settings.GOAL_INPUT+settings.GOAL_TYPE],settings.GROUP_LABELS_INSTABILITY[settings.GOAL_INPUT+settings.GOAL_TYPE])
    settings.CODES = copy.deepcopy(R[0])
    settings.LABELS = copy.deepcopy(R[1])
    settings.FEATURES = copy.deepcopy(R[2]) 
    
    for I in settings.INPUTS:
        for T in settings.TARGET_TYPE:
            settings.GROUP_LABELS[I+T] = normalize_labels(settings.GROUP_LABELS[I+T])


    settings.LABELS = normalize_labels(settings.LABELS)
    normalize_feature_numalize()
 
    # optional function
    # to save time can be commented
    augment_goal_features()