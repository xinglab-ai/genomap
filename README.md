# Genomap creates images from gene expression data and offers high-performance dimensionality reduction and visualization, data clustering, classification and regression, gene signature extraction, multi-omics data integration, and trajectory analysis 

Genomap is an entropy-based cartography strategy to contrive the high dimensional gene expression data into a configured image format with explicit integration of the genomic interactions. This unique cartography casts the gene-gene interactions into a spatial configuration and enables us to extract the deep genomic interaction features and discover underlying discriminative patterns of the data. For a wide variety of applications (cell clustering and recognition, gene signature extraction, single-cell data integration, cellular trajectory analysis, dimensionality reduction, and visualization), genomap drastically improves the accuracy of data analyses as compared to state-of-the-art techniques.

## How to use genomap

The easiest way to start with genomap is to install it from pypi using 

```python
pip install genomap
```
The data should be in cell (row) x gene (column) format. Genomap construction needs only one parameter: the size of the genomap (row and column number). The row and column number can be any number starting from 1. You can create square or rectangular genomaps. The number of genes in your dataset should be less than or equal to the number of pixels in the genomap. Genomap construction is very fast and you should get the genomaps within a few seconds. 

Please run our Code-Ocean capsules (https://codeocean.com/capsule/4321565/tree/v1 and https://codeocean.com/capsule/6967747/tree/v1) to create the results in a single click. Please check the environment section of the Code Ocean capsules if you face any issues with the packages.

## Sample data

To run the example codes below, you will need to download data files from [here](https://drive.google.com/drive/folders/1xq3bBgVP0NCMD7bGTXit0qRkL8fbutZ6?usp=drive_link).

## Example codes

### Example 1 - Construct a genomap

```python
import pandas as pd # Please install pandas and matplotlib before you run this example
import matplotlib.pyplot as plt
import scipy
import genomap as gp

data = pd.read_csv('TM_data.csv', header=None, delim_whitespace=False)
colNum=31 # Column number of genomap
rowNum=31 # Row number of genomap

dataNorm=scipy.stats.zscore(data,axis=0,ddof=1) # Normalization of the data

genoMaps=gp.construct_genomap(dataNorm,rowNum,colNum) # Construction of genomaps

findI=genoMaps[0,:,:,:]

plt.figure(1) # Plot the first genomap
plt.imshow(findI, origin = 'lower', extent = [0, 10, 0, 10], aspect = 1)
plt.title('Genomap of a cell from TM dataset')
```

### Example 2 - Try genoVis for data visualization and clustering

```python
import scipy.io as sio
import numpy as np
import metrics
from genomap.genoVis import compute_genoVis
from genomap.genoTraj import compute_genoTraj
from genomap.genoMOI import compute_genoMOI

data = pd.read_csv('TM_data.csv', header=None,
                   delim_whitespace=False)
data=data.values
gt_data = sio.loadmat('GT_TM.mat')
y = np.squeeze(gt_data['GT'])
n_clusters = len(np.unique(y))

resVis=compute_genoVis(data,n_clusters=n_clusters, colNum=33,rowNum=33)
# Use resVis=compute_genoVis(data, colNum=32,rowNum=32), if you do not know the number
# of classes in the data

resVisEmb=resVis[0] # Dimensionality reduction and visualization result
clusIndex=resVis[1] # Clustering result

plt.figure(figsize=(15, 10))
plt.rcParams.update({'font.size': 28})    
h1=plt.scatter(resVisEmb[:, 0], resVisEmb[:, 1], c=y,cmap='jet', marker='o', s=18)     
plt.xlabel('genoVis1')
plt.ylabel('genoVis2')
plt.tight_layout()
plt.colorbar(h1)

# Print clustering accuracy metrics
print('acc=%.4f, nmi=%.4f, ari=%.4f' % (metrics.acc(y, clusIndex), metrics.nmi(y, clusIndex), metrics.ari(y, clusIndex)))
```

### Example 3 - Try genoDR for dimensionality reduction

```python
import scipy.io as sio
import numpy as np
from genoDimReduction import compute_genoDimReduction
import matplotlib.pyplot as plt
import umap

dx = sio.loadmat('../data/reducedData_divseq.mat')
data=dx['X']
gt_data = sio.loadmat('../data/GT_divseq.mat')
y = np.squeeze(gt_data['GT'])
n_clusters = len(np.unique(y))

resDR=compute_genoDimReduction(data,n_clusters=n_clusters, colNum=33,rowNum=33)
#resDR=compute_genoDimReduction(data, colNum=33,rowNum=33) # if you dont know the number
# of classes in the data
embedding2D = umap.UMAP(n_neighbors=30,min_dist=0.3,n_epochs=200).fit_transform(resDR)

plt.figure(figsize=(15, 10))
plt.rcParams.update({'font.size': 28})    
h1=plt.scatter(embedding2D[:, 0], embedding2D[:, 1], c=y,cmap='jet', marker='o', s=18)      #  ax = plt.subplot(3, n, i + 1*10+1)
plt.xlabel('genoVis1')
plt.ylabel('genoVis2')
plt.tight_layout()
plt.colorbar(h1)
```

### Example 4 - Try genoTraj for cell trajectory analysis

```python
# Load data
dx = sio.loadmat('organoidData.mat')
data=dx['X3']
gt_data = sio.loadmat('cellsPsudo.mat')
Y_time = np.squeeze(gt_data['newGT'])

# Apply genoTraj for embedding showing cell trajectories
outGenoTraj=compute_genoTraj(data)

plt.figure(figsize=(15, 10))
plt.rcParams.update({'font.size': 28})    
h1=plt.scatter(outGenoTraj[:, 0], outGenoTraj[:, 1], c=Y_time,cmap='jet', marker='o', s=18)      #  ax = plt.subplot(3, n, i + 1*10+1)
plt.xlabel('genoTraj1')
plt.ylabel('genoTraj2')
plt.tight_layout()
plt.colorbar(h1)

```

### Example 5 - Try genoMOI for multi-omic data integration

```python

# Load datasets
dx = sio.loadmat('dataBaronX.mat')
data=dx['dataBaron']
dx = sio.loadmat('dataMuraroX.mat')
data2=dx['dataMuraro']
dx = sio.loadmat('dataScapleX.mat')
data3=dx['dataScaple']
dx = sio.loadmat('dataWangX.mat')
data4=dx['dataWang']
dx = sio.loadmat('dataXinX.mat')
data5=dx['dataXin']
# Load class and batch labels
dx = sio.loadmat('classLabel.mat')
y = np.squeeze(dx['classLabel'])
dx = sio.loadmat('batchLabel.mat')
ybatch = np.squeeze(dx['batchLabel'])

# Apply genoMOI
resVis=compute_genoMOI(data, data2, data3, data4, data5, colNum=44, rowNum=44)

# Visualize the integrated data using UMAP
embedding = umap.UMAP(n_neighbors=30,min_dist=0.3,n_epochs=200).fit_transform(resVis) 

plt.figure(figsize=(15, 10))
plt.rcParams.update({'font.size': 28})    
h1=plt.scatter(embedding[:, 0], embedding[:, 1], c=y,cmap='jet', marker='o', s=18)     
plt.xlabel('UMAP')
plt.ylabel('UMAP2')
plt.tight_layout()
plt.colorbar(h1)
```

### Example 6 - Try genoSig for finding gene signatures for cell/data classes

```python
import numpy as np
import scipy.io as sio
from util_Sig import createGenomap_for_sig
import pandas as pd
from compute_genoSig import genoSig

# Load data
dx = sio.loadmat('../data/reducedData_divseq.mat')
data=dx['X']
# Load data labels
label = pd.read_csv('../data/groundTruth_divseq.csv',header=None)
# Load gene names corresponding to the columns of the data
gene_names = ['Gene_' + str(i) for i in range(1, data.shape[1]+1)]
gene_names=np.array(gene_names)

# The cell classes for which gene signatures will be computed
userPD = np.array(['DG'])

colNum=32 # genomap column number
rowNum=32 # genomap row number
# Create genomaps
genoMaps,gene_namesRe,T=createGenomap_for_sig(data,gene_names,rowNum,colNum)
# compute the gene signatures
result=genoSig(genoMaps,T,label,userPD,gene_namesRe, epochs=50)

print(result.head())  
```

### Example 7 - Try genoClassification for tabular data classification

```python
import pandas as pd
import numpy as np
import scipy.io as sio
from genoClassification import genoClassification
from util_genoClassReg import select_random_values


# First, we load the TM data. Data should be in cells X genes format, 
data = pd.read_csv('TM_data.csv', header=None,
                   delim_whitespace=False)

# Creation of genomaps
# Selection of row and column number of the genomaps 
# To create square genomaps, the row and column numbers are set to be the same.
colNum=33 
rowNum=33

# Load ground truth cell labels of the TM dataset
gt_data = sio.loadmat('GT_TM.mat')
GT = np.squeeze(gt_data['GT'])
GT=GT-1 # to ensure the labels begin with 0 to conform with PyTorch

# Select 80% of data randomly for training and others for testing
indxTrain, indxTest= select_random_values(start=0, end=GT.shape[0], perc=0.8)
groundTruthTest = GT[indxTest-1]

training_data=data.values[indxTrain-1]
training_labels=GT[indxTrain-1]
test_data=data.values[indxTest-1]

est=genoClassification(training_data, training_labels, test_data, rowNum=rowNum, colNum=colNum, epoch=150)

print('Classification accuracy of genomap+genoNet:'+str(np.sum(est==groundTruthTest) / est.shape[0]))  
```


### Example 8 - Try genoRegression for tabular data regression

```python
import pandas as pd
import numpy as np
import scipy.io as sio
from genoRegression import genoRegression
from sklearn.metrics import mean_squared_error
from util_genoClassReg import select_random_values

# Load data and labels
dx = sio.loadmat('../data/organoidData.mat')
data=dx['X3']
gt_data = sio.loadmat('../data/GT_Org.mat')
Y_time = np.squeeze(gt_data['GT'])
Y_time = Y_time - 1 # to ensure the labels begin with 0 to conform with PyTorch

# Select 80% of data randomly for training and others for testing
indxTrain, indxTest= select_random_values(start=0, end=Y_time.shape[0], perc=0.8)
groundTruthTest = Y_time[indxTest-1]
training_data=data[indxTrain-1]
training_labels=Y_time[indxTrain-1]
test_data=data[indxTest-1]

# Run genoRegression
est=genoRegression(training_data, training_labels, test_data, rowNum=40, colNum=40, epoch=200)

# Calculate MSE
mse = mean_squared_error(groundTruthTest, est)
print(f'MSE: {mse}') 
```

# Citation

If you use the genomap code, please cite our Nature Communications paper: https://www.nature.com/articles/s41467-023-36383-6

Islam, M.T., Xing, L. Cartography of Genomic Interactions Enables Deep Analysis of Single-Cell Expression Data. Nat Commun 14, 679 (2023). https://doi.org/10.1038/s41467-023-36383-6



