# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 18:25:32 2023

@author: Windows
"""

import numpy as np
from genomap.genomapT import construct_genomap_returnT
# suppose you have a feature matrix X
# X = np.array(...)

def select_n_features(X,n):
# calculate the variance of each feature
    variances = np.var(X, axis=0)

# get the indices of the features sorted by variance (in descending order)
    indices = np.argsort(variances)[::-1]

# select the indices of the top 10 most variable features
    top_n_indices = indices[:n]

# select the top 10 most variable features
    X_top_n = X[:, top_n_indices]
    return X_top_n,top_n_indices


def createGenomap_for_sig(data,gene_names,rowNum=32,colNum=32):

    """
    Returns the constructed genomaps


    Parameters
    ----------
    data : ndarray, shape (cellNum, geneNum)
         gene expression data in cell X gene format. Each row corresponds
         to one cell, whereas each column represents one gene
    gene_names: numpy array, shape (1, geneNum)
        name of the genes corresponding to the columns
    rowNum : int, 
         number of rows in a genomap
    colNum : int,
         number of columns in a genomap

    Returns
    -------
    genomaps : ndarray, shape (rowNum, colNum, zAxisDimension, cell number)
           genomaps are formed for each cell. zAxisDimension is more than
           1 when 3D genomaps are created. 
    T : transfer function of the genomap construction
    gene_namesRe: selected genes in the genomaps       
    """

    nump=rowNum*colNum
    if nump<data.shape[1]:
        data,index=select_n_features(data,nump)
        gene_namesRe=gene_names[index]
      #  X_train=X_train[top_n_features]
    #ldaOutzc=scipy.stats.zscore(data, axis=0, ddof=1)    
    genoMaps,T=construct_genomap_returnT(data,rowNum,colNum,epsilon=0.0,num_iter=200)
    return genoMaps,gene_namesRe,T