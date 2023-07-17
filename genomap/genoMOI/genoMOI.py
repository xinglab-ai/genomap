# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 16:11:15 2023

@author: Md Tauhidul Islam, Ph.D., Dept. of Radiation Oncology, Stanford University
"""

from tensorflow.keras.optimizers import Adam
from genomap.utils.ConvIDEC import ConvIDEC
from sklearn.feature_selection import VarianceThreshold
from genomap.genomap import construct_genomap
import umap
from genomap.utils.gTraj_utils import nearest_divisible_by_four
from genomap.utils.utils_MOI import * 
from genomap.utils.util_Sig import select_n_features

def genoMOI(*arrays,n_clusters=None, colNum, rowNum):  

# arrays: number of arrays such as array1,array2
# n_clusters: number of data classes
# colNum and rowNum: column are rwo number of genomaps
#
# Pre-align data with bbknn
    batch_corrected_data=apply_bbknn_and_return_batch_corrected(*arrays)
    dataDX=scipy.stats.zscore(batch_corrected_data, axis=0, ddof=1) 
    
    if n_clusters==None:
        array1=arrays[0]
        adata = convertToAnnData(array1)
        # Perform Louvain clustering
        sc.pp.neighbors(adata)
        sc.tl.louvain(adata, resolution=1.0)
        # Access the cluster assignments
        cluster_labels = adata.obs['louvain']
        n_clusters = len(np.unique(cluster_labels))        
    
    resVis=extract_genoVis_features(dataDX,n_clusters=n_clusters, colNum=colNum,rowNum=rowNum)
    return resVis


def extract_genoVis_features(data,n_clusters=20, colNum=32,rowNum=32,batch_size=64,verbose=1,
                    pretrain_epochs=100,maxiter=300):
# rowNum and colNum are the row and column numbers of constructed genomaps
# n_clusters: number of data classes in the data
# batch_size: number of samples in each mini batch while training the deep neural network
# verbose: whether training progress will be shown or not
# pretrain_epochs: number of epoch for pre-training the CNN
# maxiter: number of epoch of fine-tuning training
# Construction of genomap    
    colNum=nearest_divisible_by_four(colNum)
    rowNum=nearest_divisible_by_four(rowNum)
    nump=rowNum*colNum 
    if nump<data.shape[1]:
        data,index=select_n_features(data,nump) 
    genoMaps=construct_genomap(data,rowNum,colNum,epsilon=0.0,num_iter=200)

# Deep learning-based dimensionality reduction and clustering
    optimizer = Adam()    
    model = ConvIDEC(input_shape=genoMaps.shape[1:], filters=[32, 64, 128, 32], n_clusters=n_clusters)
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
    #embedding2D = umap.UMAP(n_neighbors=30,min_dist=0.3,n_epochs=200).fit_transform(feat_DNN)
    return feat_DNN




