# Genomap creates images from gene expression data

Genomap is an entropy-based cartography strategy to contrive the high dimensional gene expression data into a configured image format with explicit integration of the genomic interactions. This unique cartography casts the gene-gene interactions into a spatial configuration and enables us to extract the deep genomic interaction features and discover underlying discriminative patterns of the data. For a wide variety of applications (cell clustering and recognition, gene signature extraction, single-cell data integration, cellular trajectory analysis, dimensionality reduction, and visualization), genomap drastically improves the accuracy of data analyses as compared to state-of-the-art techniques.

## Required packages

scipy, scikit-learn, pot, numpy

If you face any issues with packages, please check the environment section of our Code-Ocean capsule (https://doi.org/10.24433/CO.0640398.v1), where you can check the package versions.

## How to use genomap

The easiest way to start with genomap is to install it from pypi using 

```python
pip install genomap
```
The data should be in cell (row) x gene (column) format. Genomap construction needs only one parameter: the size of the genomap (row and column number). The row and column number can be any number starting from 1. You can create square or rectangular genomaps. The number of genes in your dataset should be less than or equal to the number of pixels in the genomap. Genomap construction is very fast and you should get the genomaps within a few seconds.

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

### Example 2 - Try out genoVis, genoTraj and genoMOI

```python
import scipy.io as sio
import numpy as np
from genomap.genoVis import compute_genoVis
from genomap.genoTraj import compute_genoTraj
from genomap.genoMOI import compute_genoMOI

dx = sio.loadmat('reducedData_divseq.mat')
data=dx['X']
gt_data = sio.loadmat('GT_divseq.mat')
y = np.squeeze(gt_data['GT'])
n_clusters = len(np.unique(y))


resVis=compute_genoVis(data,n_clusters=n_clusters, colNum=33,rowNum=33)

dx = sio.loadmat('organoidData.mat')
data=dx['X3']
gt_data = sio.loadmat('cellsPsudo.mat')
Y_time = np.squeeze(gt_data['newGT'])


outGenoTraj=compute_genoTraj(data)

dx = sio.loadmat('MOIdata/dataBaronX.mat')
data=dx['dataBaron']
dx = sio.loadmat('MOIdata/dataMuraroX.mat')
data2=dx['dataMuraro']
dx = sio.loadmat('MOIdata/dataScapleX.mat')
data3=dx['dataScaple']
dx = sio.loadmat('MOIdata/dataWangX.mat')
data4=dx['dataWang']
dx = sio.loadmat('MOIdata/dataXinX.mat')
data5=dx['dataXin']

resVis=compute_genoMOI(data, data2, data3, data4, data5, colNum=44, rowNum=44)
```

# Citation

If you use the genomap code, please cite our Nature Communications paper: https://www.nature.com/articles/s41467-023-36383-6

Islam, M.T., Xing, L. Cartography of Genomic Interactions Enables Deep Analysis of Single-Cell Expression Data. Nat Commun 14, 679 (2023). https://doi.org/10.1038/s41467-023-36383-6




