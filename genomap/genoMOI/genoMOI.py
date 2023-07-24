# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 16:11:15 2023

@author: Md Tauhidul Islam, Ph.D., Dept. of Radiation Oncology, Stanford University
"""

from tensorflow.keras.optimizers import Adam
import umap
import numpy as np
import scanorama
import phate
import scanpy as sc
import scipy.io
import pandas as pd
import anndata as ad
from sklearn.feature_selection import VarianceThreshold
from genomap.utils.ConvIDEC import ConvIDEC
from genomap.genomap import construct_genomap
from genomap.utils.class_discriminative_opt import ClassDiscriminative_OPT
from genomap.utils.gTraj_utils import nearest_divisible_by_four, compute_cluster_distances
from genomap.utils.utils_MOI import * 
from genomap.utils.util_Sig import select_n_features

def selectNdimIDX(*args, N):
    data_selected_list = []

    for data in args:
        gvar = np.var(data, axis=0)
        varidx = np.argsort(gvar)[::-1]  # Sort indices in descending order
        var500idx = varidx[:N]

        var500idxS = np.sort(var500idx)
        data_selected = data[:, var500idxS]
        data_selected_list.append(data_selected)

    return data_selected_list, var500idxS

def genoMOIvis(*arrays, n_clusters=None, n_dim=32, colNum=32, rowNum=32, epoch=100, prealign_method='scanorama'):

# arrays: a number of arrays such as array1, array2 from different sources
# n_clusters: number of data classes
# n_dim: number of the dimension in returned integrated data
# colNum and rowNum: column and row number of genomaps
# # Returns the embedding, estimated cluster labels, and intgerated data
    # Pre-align data with bbknn/scanorama
    if (prealign_method=='scanorama'):
        batch_corrected_data=apply_scanorama_and_return_batch_corrected(*arrays,n_dim=colNum*rowNum)
    else:
        batch_corrected_data=apply_bbknn_and_return_batch_corrected(*arrays)
    dataDX=scipy.stats.zscore(batch_corrected_data, axis=0, ddof=1) 
    #dataDX=batch_corrected_data
    if n_clusters==None:
        array1=arrays[0]
        adata = convertToAnnData(array1)
        # Perform Louvain clustering
        sc.pp.neighbors(adata)
        sc.tl.louvain(adata, resolution=1.0)
        # Access the cluster assignments
        cluster_labels = adata.obs['louvain']
        n_clusters = len(np.unique(cluster_labels))        
    
    embedding, y_pred, feat_DNN=extract_genoVis_features(dataDX, n_clusters=n_clusters, n_dim=n_dim, colNum=colNum,rowNum=rowNum, 
                                               pretrain_epochs=epoch)    

    return embedding, y_pred, feat_DNN

def genoMOItraj(*arrays, n_clusters=None, n_dim=32, colNum=32, rowNum=32, epoch=100, prealign_method='scanorama'):

# arrays: a number of arrays such as array1, array2 from different sources
# n_clusters: number of data classes
# n_dim: number of the dimension in returned integrated data
# colNum and rowNum: column and row number of genomaps
# # Returns the embedding, estimated cluster labels, and intgerated data
    # Pre-align data with bbknn/scanorama
    if (prealign_method=='scanorama'):
        batch_corrected_data=apply_scanorama_and_return_batch_corrected(*arrays,n_dim=colNum*rowNum)
    else:
        batch_corrected_data=apply_bbknn_and_return_batch_corrected(*arrays)
    dataDX=scipy.stats.zscore(batch_corrected_data, axis=0, ddof=1) 
    #dataDX=batch_corrected_data
    if n_clusters==None:
        array1=arrays[0]
        adata = convertToAnnData(array1)
        # Perform Louvain clustering
        sc.pp.neighbors(adata)
        sc.tl.louvain(adata, resolution=1.0)
        # Access the cluster assignments
        cluster_labels = adata.obs['louvain']
        n_clusters = len(np.unique(cluster_labels))        
    
    embedding, y_pred, feat_DNN=extract_genoVis_features(dataDX, n_clusters=n_clusters, n_dim=n_dim, colNum=colNum,rowNum=rowNum, 
                                               pretrain_epochs=epoch) 
    
    outGenoTraj=apply_genoTraj(dataDX,y_pred)

    return embedding, y_pred, feat_DNN

def apply_genoTraj(data, y_pred):
    Kdis=compute_cluster_distances(data,y_pred);

# Kdis = normalize(Kdis, axis=1, norm='l1')
# Kdis=1-Kdis
    n_clusters=len(np.unique(y_pred))
    num_comp=n_clusters-1
    # Second optimization (class-discriminative optimization)
    clf = ClassDiscriminative_OPT(n_components=num_comp)
    clf.fit(Kdis, y_pred)
    ccifOut = clf.transform(Kdis)
    ccifOutzc = scipy.stats.zscore(ccifOut, axis=0, ddof=1) # Z-score of the CCIF 


    phate_op = phate.PHATE()
    data_phate = phate_op.fit_transform(ccifOutzc)
    return data_phate

def extract_genoVis_features(data,n_clusters=20, n_dim=32, colNum=32, rowNum=32, batch_size=64, verbose=1,
                    pretrain_epochs=100, maxiter=300):
# rowNum and colNum are the row and column numbers of constructed genomaps
# n_clusters: number of data classes in the data
# batch_size: number of samples in each mini batch while training the deep neural network
# verbose: whether training progress will be shown or not
# pretrain_epochs: number of epoch for pre-training the CNN
# maxiter: number of epoch of fine-tuning training
# Returns the embedding, estimated cluster labels, and intgerated data 
# Construction of genomap    
    colNum=nearest_divisible_by_four(colNum)
    rowNum=nearest_divisible_by_four(rowNum)
    nump=rowNum*colNum 
    if nump<data.shape[1]:
        data,index=select_n_features(data,nump) 
    genoMaps=construct_genomap(data,rowNum,colNum,epsilon=0.0,num_iter=200)

# Deep learning-based dimensionality reduction and clustering
    optimizer = Adam()    
    model = ConvIDEC(input_shape=genoMaps.shape[1:], filters=[32, 64, 128, n_dim], n_clusters=n_clusters)
    model.compile(optimizer=optimizer, loss=['kld', 'mse'], loss_weights=[0.1, 1.0])
    pretrain_optimizer ='adam'
    update_interval=50
    model.pretrain(genoMaps, y=None, optimizer=pretrain_optimizer, epochs=pretrain_epochs, batch_size=batch_size,
                        verbose=verbose)

    y_pred = model.fit(genoMaps, y=None, maxiter=maxiter, batch_size=batch_size, update_interval=update_interval,
                       )
    y_pred = model.predict_labels(genoMaps)
    # Extract DNN features
    feat_DNN=model.extract_features(genoMaps)
    feat_DNN=scipy.stats.zscore(feat_DNN, axis=0, ddof=1)
    embedding2D = umap.UMAP(n_neighbors=30,min_dist=0.3,n_epochs=200).fit_transform(feat_DNN)
    return embedding2D, y_pred, feat_DNN

def select_highly_variable_genes_top_genes(adata,n_top_genes=2500):
    # Select_highly_variable_genes of the data
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat_v3')
    adata = adata[:, adata.var.highly_variable]
    return adata

def nonrmalize_data(adata,target_sum=1e4):
    # Normalize the data
    sc.pp.normalize_total(adata, target_sum)
    return adata

def apply_scanorama_and_return_batch_corrected(*arrays,n_dim=50):    
    adatas = []
    for array in arrays:
        adata = convertToAnnData(array)
        sc.pp.filter_genes(adata, min_cells=0)
        sc.pp.scale(adata)        
        adatas.append(adata)
    # Concatenate the AnnData objects
    merged_adata = ad.concat(adatas, join='outer', label="batch")
    if array.shape[1]>1000:
        ngene=1000
    else:
        ngene=array.shape[1]            
    # Apply bbknn for batch correction
    sc.pp.highly_variable_genes(merged_adata,n_top_genes=ngene)
    sc.pp.pca(merged_adata, n_comps=30, use_highly_variable=True, svd_solver='arpack')   

        
    scanorama.integrate_scanpy(adatas, dimred = n_dim)    
    # Get all the integrated matrices.
    scanorama_int = [ad.obsm['X_scanorama'] for ad in adatas]

    # make into one matrix.
    all_s = np.concatenate(scanorama_int)
    
    # Extract the batch-corrected data
    batch_corrected_data = all_s
    # Return the batch-corrected data as a NumPy array
    return np.array(batch_corrected_data)    

def apply_bbknn_and_return_batch_corrected(*arrays):    
    adatas = []
    for array in arrays:
        adata = convertToAnnData(array)
        sc.pp.filter_genes(adata, min_cells=0)
        sc.pp.scale(adata)        
        adatas.append(adata)
    # Concatenate the AnnData objects
    merged_adata = ad.concat(adatas, join='outer', label="batch")
    if array.shape[1]>1000:
        ngene=1000
    else:
        ngene=array.shape[1]            
    # Apply bbknn for batch correction
    sc.pp.highly_variable_genes(merged_adata,n_top_genes=ngene)
    sc.pp.pca(merged_adata, n_comps=50, use_highly_variable=True, svd_solver='arpack')
    sc.external.pp.bbknn(merged_adata, batch_key='batch', neighbors_within_batch=5, n_pcs=50)    
    #sc.pp.combat(merged_adata, key='batch')
    # Extract the batch-corrected data
    batch_corrected_data = merged_adata.X
    # Return the batch-corrected data as a NumPy array
    return np.array(batch_corrected_data)
       
def convertToAnnData(data):
    # Create pseudo cell names
    cell_names = ['Cell_' + str(i) for i in range(1, data.shape[0]+1)]
    # Create pseudo gene names
    gene_names = ['Gene_' + str(i) for i in range(1, data.shape[1]+1)]
    # Create a pandas DataFrame
    df = pd.DataFrame(data, index=cell_names, columns=gene_names)
    adataMy=sc.AnnData(df)
    return adataMy

def write_numpy_array_to_mat_file(array, filename):
    # Create a dictionary with the array data
    data_dict = {"array": array}
    # Save the dictionary as a .mat file
    scipy.io.savemat(filename, data_dict)