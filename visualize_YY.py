#!/usr/bin/env python
# coding: utf-8
######################### Affinity2Vec Model Using Python ##########################
#  First author:  maha.thafar@kaust.edu.sa
   
# Done: April, 2021
 
# Description
# This script predict drug-target binding affinity (DTBA) and evaluate the model using 4 evaluation metrics

###############################################################################

# all needed packages
import pandas as pd
from copy import deepcopy
import math as math
import numpy as np
from math import exp
# import argparse
import json,csv, pickle
import itertools,collections
import matplotlib.pyplot as plt

# ML packages
from sklearn.metrics import *
from sklearn.metrics import mean_squared_error, r2_score
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
# from joblib import Parallel, delayed
# import multiprocessing
import xgboost as xgb

# Similarity and normalization packages
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.preprocessing import MinMaxScaler, minmax_scale
import scipy
from scipy import *

# DL Keras
from keras.preprocessing.sequence import skipgrams
from keras.layers import Embedding, Input, Reshape, Dense, Concatenate
from keras.models import Sequential, Model

# Import my files
from training_functions import *
from pathScores_functions import *
from evaluation import *

######################################## START MAIN #########################################
#############################################################################################

## Affinity2Vec_Hybrid source code for the best results using Davis Dataset

def main():

    # get the parameters from the user
        # args = parse_args()

    # ## Davis Dataset
    # ### Drug No. 68 --- Protein No. 442,  Known Interaction  30,056
    # ## Read the data...
    DrugID = np.genfromtxt("Input/Davis/drug_PubChem_CIDs.txt",delimiter='\n', dtype=str)
    ProteinID = np.genfromtxt("Input/Davis/target_gene_names.txt", delimiter='\n',dtype=str)

    DrPr_file = "Input/Davis/drug-target_interaction_affinities_Kd__Davis_et_al.2011v1.txt"
    DrPr_Matrix = np.loadtxt(DrPr_file,dtype=str ,skiprows=0)

    Dsim_file = "Input/Davis/drug-drug_similarities_2D.txt"
    Dr_SimM = np.loadtxt(Dsim_file, dtype=float ,skiprows=0)

    Psim_file = "Input/Davis/target-target_similarities_WS.txt"
    Pr_SimM = np.loadtxt(Psim_file, delimiter=" ",dtype=float ,skiprows=0)

    ProteinID = pd.DataFrame(ProteinID)
    ProteinID['Protein NO.'] = ProteinID.index+1

    #ProteinID = ProteinID.reset_index(level = 0, column='Protein NO.', inplace=False)
    ProteinID.rename(columns={0:'Protein Name'},  inplace=True)
    ProteinID['Protein NO.'] = ProteinID['Protein NO.'].astype(str)

    ProteinNO = ProteinID['Protein NO.']

    Pr_SimM = normalizedMatrix(Pr_SimM)

    # create 2 dictionaries for drugs. First one the keys are their order numbers
    #the second  one the keys are their names -- same for targets
    drugID = dict([(d, i) for i, d in enumerate(DrugID)])
    targetID = dict([(t, i) for i, t in enumerate(ProteinNO)])

    AllDrPr = list(itertools.product(DrugID,ProteinNO))

    allD = list(set(DrugID))
    allT = list(set(ProteinNO))
        
    label = []
    pairX = []
    for i in range(DrPr_Matrix.shape[0]):
        for j in range(DrPr_Matrix.shape[1]):
            d = DrugID[i]
            p = ProteinNO[j]
            lab = DrPr_Matrix[i][j]
            pair = d,p
            label.append(lab)
            pairX.append(pair)


    # prepare X = pairs, Y = labels and build the random forest model
    X = np.asarray(pairX) #keys --> in jupyter same order but not all compilers!!
    Y = np.asarray(label)

    print('dimensions of all pairs', X.shape)

    Pr_SimM = keep_sim_threshold(Pr_SimM ,0.04)
    Dr_SimM = keep_sim_threshold(Dr_SimM ,0.35)

    Y_transform = np.array(Y, dtype = np.float64)
    YY = -(np.log10(Y_transform/(math.pow(10,9)))) # YY is the affinity value after transform in 1d form

    print("Primeiros valores de Y_transform (valores originais):")
    print(Y_transform[:10])  # Mostra os primeiros 10 valores

    print("\nPrimeiros valores de YY (valores transformados):")
    print(YY[:10])  # Mostra os primeiros 10 valores transformados
    
    # Criar um DataFrame para visualização
    df = pd.DataFrame({
        'Y_transform': Y_transform,
        'YY': YY
    })

    print(df.head(10))  # Mostra as primeiras 10 linhas

if __name__ == "__main__":
    main()
