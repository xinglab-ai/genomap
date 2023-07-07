# -*- coding: utf-8 -*-
"""
Created on Sun May 15 09:03:29 2022
Unsupervised genoNet
@author: anonymous
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
def findConv2dOutShape(hin,win,conv,pool=2):
    # get conv arguments
    kernel_size=conv.kernel_size
    stride=conv.stride
    padding=conv.padding
    dilation=conv.dilation

    hout=np.floor((hin+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    wout=np.floor((win+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

    if pool:
        hout/=pool
        wout/=pool
    return int(hout),int(wout)


class genoNet(nn.Module):
    # Define the convolutional neural network architecture
    def __init__(self, input_shape, class_num):
        super(genoNet, self).__init__()        
        
        input_dim=input_shape[2]
        Cin, Hin, Win = 1, input_dim, input_dim
        init_f = 8
        num_fc1 = 512        
        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3, padding=1)
        h,w=findConv2dOutShape(Hin,Win,self.conv1,pool=0)
        self.num_flatten=h*w*init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, class_num)
        self.dropout = nn.Dropout(0.25)   
        self.softmax = nn.Softmax(dim=-1)
        
 
    def forward(self, x):
        x = F.relu(self.conv1(x)); 
        x = x.contiguous().view(-1, self.num_flatten)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
    
    def forwardX(self, x):
        x = F.relu(self.conv1(x)); 
        x = x.contiguous().view(-1, self.num_flatten)
        x = F.relu(self.fc1(x))

        return x
    
def _get_device():
    # help function to detect whether the computer has a GPU
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def fit(model, dataloader, epoch, criterion, optimizer, device, trainset):
    # fit one epoch
    #
    # model: the genoNet model
    # epoch: the current epoch number
    # criterion: the loss function (by default MSE)
    # optimizer: the optimizer (by default Adam)
    # device: the device that the model is running on
    # trainset: the training dataset
    
    model.train()
    running_loss = 0.0
    counter = 0
    
    for i, data in tqdm(enumerate(dataloader),
                        total=int(len(trainset)/dataloader.batch_size), disable=True):
        counter += 1
        img = data[0]
        img = img.to(device).float()
        
        label = data[1]
        label = label.to(device).long()
        optimizer.zero_grad()
        outputs = model(img)

        loss = criterion(outputs, label)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / counter
    return epoch_loss

class geneDataset(Dataset):
    # Dataset for gene expression data
    # 
    # Attribute:
    #  data: rows denote cells and columns denote the genes. 
    def __init__(self,data,label):
        self.data = data
        self.label = label
        
    def __getitem__(self, index):
        return self.data[index,:,:,:], self.label[index]
    
    def __len__(self):
        return len(self.data)

def predict(model, dataloader, device):
    # predict the transformed data to calculate ideal distribution with trained model
    prediction_list = []
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            img = data[0]
            img = img.to(device).float()

            pred = model(img)
            prediction_list.append(pred.cpu().numpy())
    return np.concatenate(prediction_list)
    
def traingenoNet(data, labels, maxEPOCH, batchSize=16, verbose=False):
    # train the Sparse Autoencoder model
    #
    # data: training data, rows denote cells and columns denote the genes.
    # reduced_Dim: neurons in the hidden layer
    # maxEPOCH: max training EPOCH number
    # batchSize: batchSize for training
    # l2: the weight for l2 regularizer of all parameters
    # sparse_rho: the desired value for KL Divergence
    # sparse: the weight for sparse regularizer (KL Divergence loss)
    device = _get_device()
    
    trainset = geneDataset(data, labels)
    trainloader = DataLoader(trainset,
                             batch_size=batchSize,
                             shuffle=True)
    
    input_dim = data.shape[1:] # 33 x 33 x 1
    class_num, class_weight = np.unique(labels, return_counts=True)
    class_num = len(class_num)
    class_weight = torch.Tensor(np.sum(class_weight) / class_weight).to(device)
    model = genoNet(input_dim, class_num).to(device)    
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_loss = []
    for epoch in range(maxEPOCH):
        train_epoch_loss = fit(model, trainloader, epoch,
                               criterion, optimizer,device, trainset)
        if verbose:
            if epoch % 10 == 0:
                print(str(epoch) + ': ' + str(train_epoch_loss))
        train_loss.append(train_epoch_loss)
    
    
    return model

def load_genoNet(input_shape, class_num, path):
    device = torch.device('cpu')
    model = genoNet(input_shape, class_num)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model