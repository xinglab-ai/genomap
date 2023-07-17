# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:26:22 2023

@author: Windows
"""
import scanpy as sc
import numpy as np
import scipy.io
import pandas as pd
import anndata as ad


def select_highly_variable_genes_top_genes(adata,n_top_genes=2500):
    # Select_highly_variable_genes of the data
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat_v3')
    adata = adata[:, adata.var.highly_variable]
    return adata

def nonrmalize_data(adata,target_sum=1e4):
    # Normalize the data
    sc.pp.normalize_total(adata, target_sum)
    return adata

def apply_bbknn_and_return_batch_corrected(*arrays):    
    adatas = []
    for array in arrays:
        adata = convertToAnnData(array)
        sc.pp.filter_genes(adata, min_cells=0)
        sc.pp.scale(adata)        
        adatas.append(adata)
    # Concatenate the AnnData objects
    merged_adata = ad.concat(adatas, join='outer', label="batch")
    if array.shape[1]>300:
        ngene=300
    else:
        ngene=array.shape[1]            
    # Apply bbknn for batch correction
    sc.pp.highly_variable_genes(merged_adata,n_top_genes=ngene)
    sc.pp.pca(merged_adata, n_comps=30, use_highly_variable=True, svd_solver='arpack')
    sc.external.pp.bbknn(merged_adata, batch_key='batch')    
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

