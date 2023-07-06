# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 14:01:17 2023

@author: tauhi
"""

# For the tasks below, four datasets analysed in the manuscript will be automatically loaded. 
# However, you can upload your own dataset, create genomaps and train the supervised or unsupervised
# genoNet models for different tasks. 
# Our data were saved as .mat file to reduce the data size (normally .csv file needs more disk space). 
# However, .csv files can also be loaded in the way shown in the third section

# Load all necessary python packages needed for the reported analyses
# in our manuscript

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io as sio
import numpy as np
import scipy.io as sio
import genoNet as gNet
import os
import torch
from torch.utils.data import DataLoader, Dataset
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm
import sklearn
import phate
from sklearn.manifold import TSNE
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster import cluster_visualizer
from sklearn.feature_selection import VarianceThreshold

from genoNet import geneDataset,_get_device,load_genoNet,predict,traingenoNet,rescale
from genomap import construct_genomap

# For reproducibility
torch.manual_seed(1)
torch.use_deterministic_algorithms(True)


# First, we load the TM data. Data should be in cells X genes format, 
# i.e., each row should correspond to gene expressions of a cell and each column
# should represent the expression value of a specific gene. 
# The original TM data is very large with 54,865 cells and 19,791 genes. 
# We select the 1089 most variable genes from the data to reduce its size. 
# The reason behind chosing 1089 is that it is a square number (33*33)
# However, you can choose any other number. The following commented 4-lines code 
# allows one to select the most variable genes (numGene=1089 was used in our anlaysis)
# varX=np.var(dataFull,axis=0)
# idxV=np.argsort(varX)
# idxVX=np.flip(idxV)
# dataReduced=dataFull[:,idxVX[0:numGene]]

data = pd.read_csv('data/TM_data.csv', header=None,
                   delim_whitespace=False)

# Creation of genomaps
# Selection of row and column number of the genomaps 
# To create square genomaps, the row and column numbers are set to be the same.
colNum=31
rowNum=31
n=rowNum*colNum # Total number of pixels in genomaps



# When the dataset has more genes than number of pixels in the desired genomap,
# select the first n most variable genes
if n<data.shape[1]:
    # create an instance of the VarianceThreshold class
    selector = VarianceThreshold()
    # fit the selector to the data and get the indices of the top n most variable features
    var_threshold = selector.fit(data)
    top_n_indices = var_threshold.get_support(indices=True)
    top_n_features=data.columns[top_n_indices[0:n]]
    data=data[top_n_features]
# Normalization of the data
dataNorm=scipy.stats.zscore(data,axis=0,ddof=1)
# Construction of genomaps
genoMaps=construct_genomap(dataNorm,rowNum,colNum,epsilon=0.0,num_iter=200)

# Visualization of the constructed genomaps:
# The "genoMaps" array: All the constructed genomaps are saved in the array. This 
# array has four indices. The first one indexes the series of genomaps. 
# The second and third ones denote the row and column number in a genomap.
# The fourth index is introduced to facillitate the caclulation using Pytorch or Tensorflow  
# to visualize  the i-th genomap, set the first index variable to i (i=10 here) 
findI=genoMaps[10,:,:,:]
plt.figure(1)
plt.imshow(findI, origin = 'lower',  extent = [0, 10, 0, 10], aspect = 1)
plt.title('Genomap of a cell from TM dataset')



## 
# CELL CLASSIFICATION
##


# Load ground truth cell labels of the TM dataset
gt_data = sio.loadmat('data/GT_TM.mat')
GT = np.squeeze(gt_data['GT'])

# Load the indices of training and test data (70% training, 30% testing data)
index_data = sio.loadmat('data/index_TM.mat')
indxTest = np.squeeze(index_data['indxTest'])
indxTrain = np.squeeze(index_data['indxTrain'])

GT = GT - 1 # to ensure the labels begin with 0 to conform with PyTorch

# Split the data for training and testing
dataMat_CNNtrain = genoMaps[indxTrain-1,:,:,:] # to ensure the index starts at 0 to conform with
# python indexing
dataMat_CNNtest = genoMaps[indxTest-1,:,:,:]
groundTruthTest = GT[indxTest-1]
groundTruthTrain = GT[indxTrain-1]
classNum = len(np.unique(groundTruthTrain))

# Preparation of training and testing data for PyTorch computation
XTrain = dataMat_CNNtrain.transpose([0,3,1,2])
XTest = dataMat_CNNtest.transpose([0,3,1,2])
yTrain = groundTruthTrain
yTest = groundTruthTest


# Train the network in PyTorch framework
miniBatchSize = 128
net = gNet.traingenoNet(XTrain, yTrain, maxEPOCH=150, batchSize=miniBatchSize, verbose=True)

# Process the test data
testset = gNet.geneDataset(XTest, yTest)
testloader = gNet.DataLoader(testset, batch_size=miniBatchSize, shuffle=False)
device = gNet._get_device()

# Perform cell classification/reocognition
prob_test = gNet.predict(net, testloader, device)
pred_test = np.argmax(prob_test, axis=-1)

# Compute the accuracy of cell classification/reocognition
print('Classification accuracy of genomap+genoNet for TM dataset:'+str(np.sum(pred_test==yTest) / pred_test.shape[0]))



## 
# MULTI_OMIC INTEGRATION
##


# The pancreatic scRNA-seq data from 5 different technologies (Baron et al, 
# Muraro et al., Xin et al., Wang et al., and Segerstolpe et al.) are first integrated with Seurat. 
# In Seurat, the parameter 'nfeatures' (denoting  the number of features) is set
# to 2000. The resulting Seurat output is loaded here
data = sio.loadmat('data/outSeurat_pancORG.mat')
outSeurat=data['outSeurat']

# We now create a genomap for each cell in the data
# We select the nearest square number to 2000, which is 1936
# Thus the size of the genomap would be 44 by 44.
# Next, let us select data with 1936 most variable features 
numRow=44
numCol=44
varX=np.var(outSeurat,axis=0)
idxV=np.argsort(varX)
idxVX=np.flip(idxV)
outSeurat1936=outSeurat[:,idxVX[0:numRow*numCol]]

# Construction of the genomaps for the pancreatic data from the five different technologies
genoMaps=construct_genomap(outSeurat1936,numRow,numCol,epsilon=0.0,num_iter=200)


# Visualize a genomap (we show here the first one (i=0))
findI=genoMaps[0,:,:,:]
plt.figure(4)
plt.imshow(findI, origin = 'lower',  extent = [0, 10, 0, 10], aspect = 1)
plt.title('Genomap of a cell from pancreatic dataset')
# Load the data labels
gt_data = sio.loadmat('data/GT_panc.mat')
GT = np.squeeze(gt_data['GT'])

# Load the index of the training and testing data 
# Here training data are from Baron et al, Muraro et al., Xin et al., and Wang et al.
#  and the testing data is from Segerstolpe et al.
index_data = sio.loadmat('data/index_panc.mat')
indxTest = np.squeeze(index_data['indxTest'])
indxTrain = np.squeeze(index_data['indxTrain'])
GT = GT - 1 

# Prepare the data for genoNet training
dataMat_CNNtrain = genoMaps[indxTrain-1,:,:,:] # transfer from matlab indexing to python indexing
dataMat_CNNtest = genoMaps[indxTest-1,:,:,:]
groundTruthTest = GT[indxTest-1]
groundTruthTrain = GT[indxTrain-1]

XTrain = dataMat_CNNtrain[:,:,:,:].transpose([0,3,1,2])
XTest = dataMat_CNNtest.transpose([0,3,1,2])
# Make labels begin with 0
yTrain = groundTruthTrain-1
yTest = groundTruthTest-1

miniBatchSize = 128
# Train the genoNet
torch.manual_seed(0)
net = traingenoNet(XTrain, yTrain, maxEPOCH=150, batchSize=miniBatchSize, verbose=True)

# Test the genoNet
testset = geneDataset(XTest, yTest)
testloader = DataLoader(testset, batch_size=miniBatchSize, shuffle=False)
device = torch.device('cpu')
prob_test = predict(net, testloader, device)
pred_test = np.argmax(prob_test, axis=-1)

print('Label transfer accuracy of genomap+genoNet:'+ str(np.sum(pred_test==yTest) / pred_test.shape[0]))


##
# Cellular trajectory mapping
##


# As cellular trajectory mapping is an unsupervised task, we import the unsupervised genoNet. 
# The difference between a supervised and unsupervised
# genoNet is the size of the final fully connected (FFC) layer. As unsupervised genoNet
# is used for feature extraction, its FFC layer needs to have much more neurons than the
# supervised genoNet. We set the neuron number in the FFC layer of supervised
# genoNet to 100 and of unsupervised genoNet to 512. 

# import the unsupervised genoNet
from genoNetus import genoNet,geneDataset,load_genoNet,predict,traingenoNet 

# For the purpose of rapid demonstration, we have created/saved the genomaps for this dataset  
# The saved genomaps are uploaded here. 
# However, one can also create the genomaps from data/data_proto.mat by
# following the process described in the third section.

# Load the genomaps and tabular data of proto-vertebrate dataset
data = sio.loadmat('data/genoMaps_Proto.mat')
genoMaps = data['genoMaps']
dataX = sio.loadmat('data/data_proto.mat')
dataVec = dataX['X']

# Load data labels for visualization
gt_data = sio.loadmat('data/GT_proto.mat')
GrTruth = np.squeeze(gt_data['GT'])

# Use K-means++ for the initial clustering
amount_initial_centers = 10
initial_centers = kmeans_plusplus_initializer(dataVec, amount_initial_centers).initialize()
# Use x-means for clustering the data
xmeans_instance = xmeans(dataVec, initial_centers, 20)
xmeans_instance.process()
# Extract clustering results: clusters and their centers
clusters = xmeans_instance.get_clusters()
centers = xmeans_instance.get_centers()

clusIndex=np.zeros(dataVec.shape[0]);
for i in range(0,len(clusters)):
    a=(clusters[i])
    clusIndex[a]=i;
    

# Set up the training and testing data 
dataMat_CNNtrain = genoMaps # For unsupervised genoNet, all the data are used for training
dataMat_CNNtest = genoMaps

# Use the estimated labels by x-means clustering for training the unsupervised genoNet
groundTruthTest = clusIndex
groundTruthTrain = clusIndex
classNum = len(np.unique(groundTruthTrain))
XTrain = dataMat_CNNtrain[:,:,:,:].transpose([3,2,0,1])
XTest = dataMat_CNNtest.transpose([3,2,0,1])
yTrain = groundTruthTrain
yTest = groundTruthTest

# Train the genoNet
miniBatchSize = 128

# For the purpose of rapid demonstration, we have trained the genoNet model, which
# is uploaded here. But if one wants to train the 
# genoNet on his/her dataset, please uncomment the following code:
# net = traingenoNet(XTrain, yTrain, maxEPOCH=30, batchSize=miniBatchSize, verbose=True)

# Load the trained genoNet. 
net = load_genoNet([1,27,27], 20, 'data/genoNet_PHATE_ZF.pt')

# Extract genoNet features from the final fully connected layer
# for PHATE analysis. 
gx=np.reshape(XTrain,(XTrain.shape[0],1,XTrain.shape[2],XTrain.shape[3]))
device = _get_device()
t = torch.from_numpy(gx).to(device)
net=net.double()
dataAtFC=net.forwardX(t)
data=dataAtFC.cpu().detach().numpy()

# Run PHATE on the genoNet features
phate_op = phate.PHATE(random_state=1)
X_embedded = phate_op.fit_transform(data)

# Plot embeddings
plt.figure(11)
scatter=plt.scatter(X_embedded[:,0], X_embedded[:,1],s=2,c=GrTruth,cmap="jet")
plt.xlabel("PHATE1")
plt.ylabel("PHATE2")
plt.title('PHATE embedding of genoNet features')
legend1 = plt.legend(*scatter.legend_elements(),
                    loc="lower left", title="Time points")
plt.show()

# Run PHATE on the raw data and compare with result from the proposed approach
X_embedded = phate_op.fit_transform(dataVec)

# Plot embeddings
plt.figure(12)
scatter=plt.scatter(X_embedded[:,0], X_embedded[:,1],s=2,c=GrTruth,cmap="jet")
plt.xlabel("PHATE1")
plt.ylabel("PHATE2")
plt.title('PHATE embedding of raw data')
legend1 = plt.legend(*scatter.legend_elements(),
                    loc="lower left", title="Time points")
plt.show()





##
# Dimensionality reduction, visualization and clustering
##

# We use unsupervised genoNet for dimensionality reduction.
# We create the genomaps of a dataset and then train the unsupervised genoNet
# using the genomaps. We then extract features from the final fully connected layer of the genoNet, which
# have a lower dimensionality than the original data. We then apply
# t-SNE on the features to obtain the embeddings. 

# Load the genomaps and tabular data of comprehensive classification of mouse retinal bipolar cells 
data = sio.loadmat('data/genoMap_comClass.mat')
genoMaps = data['genoMaps']
dataX = sio.loadmat('data/data_comClass.mat')
dataVec = dataX['X']

# Load cell labels for visualization
gt_data = sio.loadmat('data/GT_comClass.mat')
GrTruth = np.squeeze(gt_data['GT'])

# Use k-means++ for initial clustering
amount_initial_centers = 10
initial_centers = kmeans_plusplus_initializer(dataVec, amount_initial_centers).initialize()
# Use x-means for clustering the data
xmeans_instance = xmeans(dataVec, initial_centers, 20)
xmeans_instance.process()
# Extract clustering results: clusters and their centers
clusters = xmeans_instance.get_clusters()
centers = xmeans_instance.get_centers()

clusIndex=np.zeros(dataVec.shape[0]);
for i in range(0,len(clusters)):
    a=(clusters[i])
    clusIndex[a]=i;
    

# Set up the training and testing data 
dataMat_CNNtrain = genoMaps # For unsupervised genoNet, all the data are used for training
dataMat_CNNtest = genoMaps

# Use the estimated labels for genoNet training
groundTruthTest = clusIndex
groundTruthTrain = clusIndex
classNum = len(np.unique(groundTruthTrain))

# Prepare data for training 
XTrain = dataMat_CNNtrain[:,:,:,:].transpose([3,2,0,1])
XTest = dataMat_CNNtest.transpose([3,2,0,1])
yTrain = groundTruthTrain
yTest = groundTruthTest

# Train the genoNet
miniBatchSize = 128

# If one wants to train the genoNet on his/her dataset, please uncomment the folloiwng code line
# net = traingenoNet(XTrain, yTrain, maxEPOCH=30, batchSize=miniBatchSize, verbose=True)

# Load the trained genoNet
net = load_genoNet([1,33,33], 19, 'data/genoNet_TSNE_ComClass.pt')
# Extract genoNet features from the final fully connected layer
gx=np.reshape(XTrain,(XTrain.shape[0],1,XTrain.shape[2],XTrain.shape[3]))
device = _get_device()
t = torch.from_numpy(gx).to(device)
net=net.double()
dataAtFC=net.forwardX(t)

# Run t-SNE on the genoNet features
X=dataAtFC.cpu().detach().numpy()
X_embedded = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(X)

plt.figure(13)
scatter=plt.scatter(X_embedded[:,0], X_embedded[:,1],s=2,c=GrTruth,cmap="jet")
plt.xlabel("tSNE1")
plt.ylabel("tSNE2")
plt.title('t-SNE embedding of genoNet features')
legend1 = plt.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
plt.show()


X_embedded = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(dataVec)

plt.figure(14)
scatter=plt.scatter(X_embedded[:,0], X_embedded[:,1],s=2,c=GrTruth,cmap="jet")
plt.xlabel("tSNE1")
plt.ylabel("tSNE2")
plt.title('t-SNE embedding of raw data')
legend1 = plt.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
plt.show()





















