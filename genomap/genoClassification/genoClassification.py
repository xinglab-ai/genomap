
"""
Created on Sun Jul 16 21:27:17 2023
@author: Md Tauhidul Islam, Research Scientist, Dept. of radiation Oncology, Stanford University
"""

import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
import scipy

from genomap.genomap import construct_genomap
from genomap.utils.util_Sig import select_n_features
import genomap.genoNet as gNet

def genoClassification(training_data, training_labels, test_data, rowNum=32, colNum=32, epoch=100):

    # training_data: numpy array in cellxgene shape
    # training_labels: numpy array 
    # test_data: numpy array in cellxgene shape
    # rowNum: row number of genomap
    # colNum: column number of genomap
    # epoch: epoch number for training
    
    data = np.concatenate((training_data, test_data), axis=0)
    # Construction of genomaps
    nump=rowNum*colNum 
    if nump<data.shape[1]:
        data,index=select_n_features(data,nump)
    
    dataNorm=scipy.stats.zscore(data,axis=0,ddof=1)
    genoMaps=construct_genomap(dataNorm,rowNum,colNum,epsilon=0.0,num_iter=200)    

    # Split the data for training and testing
    dataMat_CNNtrain = genoMaps[:training_labels.shape[0]] 
    dataMat_CNNtest = genoMaps[training_labels.shape[0]:]

    groundTruthTrain = training_labels
    classNum = len(np.unique(groundTruthTrain))

    # Preparation of training and testing data for PyTorch computation
    XTrain = dataMat_CNNtrain.transpose([0,3,1,2])
    XTest = dataMat_CNNtest.transpose([0,3,1,2])
    yTrain = groundTruthTrain

    # Train the network in PyTorch framework
    miniBatchSize = 128
    net = gNet.traingenoNet(XTrain, yTrain, maxEPOCH=epoch, batchSize=miniBatchSize, verbose=True)

    # # Process the test data
    yTest = np.random.rand(dataMat_CNNtest.shape[0])
    testset = gNet.geneDataset(XTest, yTest)
    testloader = gNet.DataLoader(testset, batch_size=miniBatchSize, shuffle=False)
    device = gNet.get_device()

    # Perform data classification
    prob_test = gNet.predict(net, testloader, device)
    pred_test = np.argmax(prob_test, axis=-1)
    
    return pred_test
    
    
    
    
