
"""
Created on Sun Jul 16 21:27:17 2023
@author: Md Tauhidul Islam, Research Scientist, Dept. of radiation Oncology, Stanford University
"""

import phate
import numpy as np
from genomap.utils.class_discriminative_opt import ClassDiscriminative_OPT
import scipy
from genomap.utils.gTraj_utils import compute_cluster_distances, nearest_divisible_by_four
from tensorflow.keras.optimizers import Adam
from genomap.utils.ConvIDEC import ConvIDEC
from sklearn.feature_selection import VarianceThreshold
from genomap.genomap import construct_genomap
from genomap.utils.util_Sig import select_n_features

def apply_genoTraj(data, y_pred):
# data: input data
# y_pred: predicted pseudo labels
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

def genoTraj(data,n_clusters = 33, colNum=32,rowNum=32,batch_size=64,verbose=1,
                    pretrain_epochs=100,maxiter=300):
# rowNum and colNum are the row and column numbers of constructed genomaps
# n_clusters: number of  pseudo-classes in the data, should be set as 17, 33 or 65
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

# Deep learning-based trajectory mapping
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

    dataX = scipy.stats.zscore(data, axis=0, ddof=1)
    outGenoTraj=apply_genoTraj(dataX,y_pred)
    return outGenoTraj 
