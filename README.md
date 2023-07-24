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

### Example 1 - Construct genomaps

```python
import pandas as pd # Please install pandas and matplotlib before you run this example
import matplotlib.pyplot as plt
import numpy as np
import scipy
import genomap as gp

data = pd.read_csv('TM_data.csv', header=None,
                   delim_whitespace=False)
colNum=33 # Column number of genomap
rowNum=33 # Row number of genomap

dataNorm=scipy.stats.zscore(data,axis=0,ddof=1) # Normalization of the data

genoMaps=gp.construct_genomap(dataNorm,rowNum,colNum,epsilon=0.0,num_iter=200) # Construction of genomaps

findI=genoMaps[10,:,:,:]

plt.figure(1) # Plot the first genomap
plt.imshow(findI, origin = 'lower',  extent = [0, 10, 0, 10], aspect = 1)
plt.title('Genomap of a cell from TM dataset')
plt.show()
```

### Example 2 - Try genoVis for data visualization and clustering

```python
import scipy.io as sio
import numpy as np
import pandas as pd
import genomap.genoVis as gp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import phate
import umap.umap_ as umap

data = pd.read_csv('TM_data.csv', header=None,
                   delim_whitespace=False)
data=data.values
gt_data = sio.loadmat('GT_TM.mat')
y = np.squeeze(gt_data['GT'])
n_clusters = len(np.unique(y))


resVis=gp.genoVis(data,n_clusters=n_clusters, colNum=33,rowNum=33)
# Use resVis=gp.genoVis(data, colNum=32,rowNum=32), if you dont know the number
# of classes in the data

resVisEmb=resVis[0] # Visualization result
clusIndex=resVis[1] # Clustering result

plt.figure(figsize=(15, 10))
plt.rcParams.update({'font.size': 28})    
h1=plt.scatter(resVisEmb[:, 0], resVisEmb[:, 1], c=y,cmap='jet', marker='o', s=18)      #  ax = plt.subplot(3, n, i + 1*10+1)
plt.xlabel('genoVis1')
plt.ylabel('genoVis2')
plt.tight_layout()
plt.colorbar(h1)
plt.show()

import genomap.utils.metrics as metrics
print('acc=%.4f, nmi=%.4f, ari=%.4f' % (metrics.acc(y, clusIndex), metrics.nmi(y, clusIndex), metrics.ari(y, clusIndex)))
```

### Example 3 - Try genoDR for dimensionality reduction

```python
import scipy.io as sio
import numpy as np
import genomap.genoDR as gp
import matplotlib.pyplot as plt
import umap.umap_ as umap

dx = sio.loadmat('reducedData_divseq.mat')
data=dx['X']
gt_data = sio.loadmat('GT_divseq.mat')
y = np.squeeze(gt_data['GT'])
n_clusters = len(np.unique(y))

reduced_dim=32 # Number of reduced dimension
resDR=gp.genoDR(data, n_dim=reduced_dim, n_clusters=n_clusters, colNum=33,rowNum=33) 
#resDR=gp.genoDR(data, n_dim=reduced_dim, colNum=33,rowNum=33) # if you dont know the number
# of classes in the data
embedding2D = umap.UMAP(n_neighbors=30,min_dist=0.3,n_epochs=200).fit_transform(resDR)

plt.figure(figsize=(15, 10))
plt.rcParams.update({'font.size': 28})    
h1=plt.scatter(embedding2D[:, 0], embedding2D[:, 1], c=y,cmap='jet', marker='o', s=18)      #  ax = plt.subplot(3, n, i + 1*10+1)
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.tight_layout()
plt.colorbar(h1)
plt.show()
```

### Example 4 - Try genoTraj for cell trajectory analysis

```python
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import phate
import umap.umap_ as umap
import genomap.genoTraj as gp

# Load data
dx = sio.loadmat('organoidData.mat')
data=dx['X3']
gt_data = sio.loadmat('cellsPsudo.mat')
Y_time = np.squeeze(gt_data['newGT'])

# Apply genoTraj for embedding showing cell trajectories
outGenoTraj=gp.genoTraj(data)

plt.figure(figsize=(15, 10))
plt.rcParams.update({'font.size': 28})    
h1=plt.scatter(outGenoTraj[:, 0], outGenoTraj[:, 1], c=Y_time,cmap='jet', marker='o', s=18)      #  ax = plt.subplot(3, n, i + 1*10+1)
plt.xlabel('genoTraj1')
plt.ylabel('genoTraj2')
plt.tight_layout()
plt.colorbar(h1)
plt.show()

# Comparison with PHATE
pca = PCA(n_components=100)
resPCA=pca.fit_transform(data)

phate_op = phate.PHATE()
res_phate = phate_op.fit_transform(resPCA)
    
    
plt.figure(figsize=(15, 10))
plt.rcParams.update({'font.size': 28})    
h1=plt.scatter(res_phate[:, 0], res_phate[:, 1], c=Y_time,cmap='jet', marker='o', s=18)      #  ax = plt.subplot(3, n, i + 1*10+1)
plt.xlabel('PHATE1')
plt.ylabel('PHATE2')
plt.tight_layout()
plt.colorbar(h1)
plt.show()
```

### Example 5 - Try genoMOI for multi-omic data integration

```python
import scanpy as sc
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import pandas as pd
import genomap.genoMOI as gp

# Load five different pancreatic datasets
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

# Apply genomap-based multi omic integration and visualize the integrated data with local structure for cluster analysis
# returns 2D visualization, cluster labels, and intgerated data
resVis,cli,int_data=gp.genoMOIvis(data, data2, data3, data4, data5, colNum=12, rowNum=12, n_dim=32, epoch=10, prealign_method='scanorama')


plt.figure(figsize=(15, 10))
plt.rcParams.update({'font.size': 28})    
h1=plt.scatter(resVis[:, 0], resVis[:, 1], c=y,cmap='jet', marker='o', s=18)      #  ax = plt.subplot(3, n, i + 1*10+1)
plt.xlabel('genoVis1')
plt.ylabel('genoVis2')
plt.tight_layout()
plt.colorbar(h1)
plt.show()

plt.figure(figsize=(15, 10))
plt.rcParams.update({'font.size': 28})    
h1=plt.scatter(resVis[:, 0], resVis[:, 1], c=ybatch,cmap='jet', marker='o', s=18)      #  ax = plt.subplot(3, n, i + 1*10+1)
plt.xlabel('genoVis1')
plt.ylabel('genoVis2')
plt.tight_layout()
plt.colorbar(h1)
plt.show()
```

```python
# Apply genomap-based multi omic integration and visualize the integrated data with global structure for trajectory analysis

# returns 2D embedding, cluster labels, and intgerated data
resTraj,cli,int_data=gp.genoMOItraj(data, data2, data3, data4, data5, colNum=12, rowNum=12, n_dim=32, epoch=10, prealign_method='scanorama')


plt.figure(figsize=(15, 10))
plt.rcParams.update({'font.size': 28})    
h1=plt.scatter(resTraj[:, 0], resTraj[:, 1], c=y,cmap='jet', marker='o', s=18)      #  ax = plt.subplot(3, n, i + 1*10+1)
plt.xlabel('genoTraj1')
plt.ylabel('genoTraj2')
plt.tight_layout()
plt.colorbar(h1)
plt.show()

plt.figure(figsize=(15, 10))
plt.rcParams.update({'font.size': 28})    
h1=plt.scatter(resTraj[:, 0], resTraj[:, 1], c=ybatch,cmap='jet', marker='o', s=18)      #  ax = plt.subplot(3, n, i + 1*10+1)
plt.xlabel('genoTraj1')
plt.ylabel('genoTraj2')
plt.tight_layout()
plt.colorbar(h1)
plt.show()
```

### Example 6 - Try genoAnnotate for cell annotation

```python
import scanpy as sc
import pandas as pd
import genomap.genoAnnotate as gp
import matplotlib.pyplot as plt
#Load the PBMC dataset
adata = sc.read_10x_mtx("./pbmc3k_filtered_gene_bc_matrices/")

# Input: adata: annData containing the raw gene counts
# tissue type: e.g. Immune system,Pancreas,Liver,Eye,Kidney,Brain,Lung,Adrenal,Heart,Intestine,Muscle,Placenta,Spleen,Stomach,Thymus 
 
adataP=gp.genoAnnotate(adata,species="human", tissue_type="Immune system")
cell_annotations=adataP.obs['cell_type'].values # numpy array containing the
# cell annotations

# Compute t-SNE
sc.tl.tsne(adataP)
# Create a t-SNE plot colored by cell type labels
cell_annotations=adataP.obs['cell_type']
sc.pl.tsne(adataP, color='cell_type')
```

### Example 7 - Try genoSig for finding gene signatures for cell/data classes

```python
import numpy as np
import scipy.io as sio
from genomap.utils.util_Sig import createGenomap_for_sig
import pandas as pd
import genomap.genoSig as gp

# Load data
dx = sio.loadmat('reducedData_divseq.mat')
data=dx['X']
# Load data labels
label = pd.read_csv('groundTruth_divseq.csv',header=None)
# Load gene names corresponding to the columns of the data
# Here we create artificial gene names as Gene_1, Gene_2. You can upload your gene sets
gene_names = ['Gene_' + str(i) for i in range(1, data.shape[1]+1)]
gene_names=np.array(gene_names)

# The cell classes for which gene signatures will be computed
userPD = np.array(['DG'])

colNum=32 # genomap column number
rowNum=32 # genomap row number
# Create genomaps
genoMaps,gene_namesRe,T=createGenomap_for_sig(data,gene_names,rowNum,colNum)
# compute the gene signatures
result=gp.genoSig(genoMaps,T,label,userPD,gene_namesRe, epochs=50)

print(result.head())
```

### Example 8 - Try genoClassification for tabular data classification

```python
import pandas as pd
import numpy as np
import scipy.io as sio
import genomap.genoClassification as gp
from genomap.utils.util_genoClassReg import select_random_values

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

# Select 80% data randomly for training and others for testing
indxTrain, indxTest= select_random_values(start=0, end=GT.shape[0], perc=0.8)
groundTruthTest = GT[indxTest-1]

training_data=data.values[indxTrain-1]
training_labels=GT[indxTrain-1]
test_data=data.values[indxTest-1]

est=gp.genoClassification(training_data, training_labels, test_data, rowNum=rowNum, colNum=colNum, epoch=150)

print('Classification accuracy of genomap approach:'+str(np.sum(est==groundTruthTest) / est.shape[0]))  
```

### Example 9 - Try genoRegression for tabular data regression

```python
import pandas as pd
import numpy as np
import scipy.io as sio
import genomap.genoRegression as gp
from sklearn.metrics import mean_squared_error
from genomap.utils.util_genoClassReg import select_random_values

# Load data and labels
dx = sio.loadmat('organoidData.mat')
data=dx['X3']
gt_data = sio.loadmat('GT_Org.mat')
Y_time = np.squeeze(gt_data['GT'])
Y_time = Y_time - 1 # to ensure the labels begin with 0 to conform with PyTorch

# Select 80% data randomly for training and others for testing
indxTrain, indxTest= select_random_values(start=0, end=Y_time.shape[0], perc=0.8)
groundTruthTest = Y_time[indxTest-1]
training_data=data[indxTrain-1]
training_labels=Y_time[indxTrain-1]
test_data=data[indxTest-1]

# Run genoRegression
est=gp.genoRegression(training_data, training_labels, test_data, rowNum=40, colNum=40, epoch=200)

# Calculate MSE
mse = mean_squared_error(groundTruthTest, est)
print(f'MSE: {mse}')
```

# Citation

If you use the genomap code, please cite our Nature Communications paper: https://www.nature.com/articles/s41467-023-36383-6

Islam, M.T., Xing, L. Cartography of Genomic Interactions Enables Deep Analysis of Single-Cell Expression Data. Nat Commun 14, 679 (2023). https://doi.org/10.1038/s41467-023-36383-6



