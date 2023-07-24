# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 15:18:43 2023

@author: Md Tauhidul Islam
# This code is inspired by scType (https://github.com/IanevskiAleksandr/sc-type)
# We are in the process of using image matching techique for further enhancement
# of the cell annotation
"""


from genomap.genotype import *
import scanpy as sc

def genoAnnotate(adata, species, tissue_type, database=None):
    # Input: adata: annData containing the raw gene counts
    # species :'human' or 'mouse'
    # tissue type: e.g. Immune system,Pancreas,Liver,Eye,Kidney,Brain,Lung,Adrenal,Heart,Intestine,Muscle,Placenta,Spleen,Stomach,Thymus 
    # database: User can select his/her own database in excel format

    # Database file
    if database==None:
        database = "https://raw.githubusercontent.com/xinglab-ai/self-consistent-expression-recovery-machine/master/demo/data/genoANN_db.xlsx";        
    
    # Filter cells
    sc.pp.filter_cells(adata, min_genes=200)
    # Normalize data
    adata.raw = adata
    sc.pp.normalize_total(adata, target_sum=10000)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable']]
    # Scale data and run PCA
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata)
    
    # Prepare positive and negative gene sets
    result = gene_sets_prepare(database, tissue_type, species)
    gs = result['gs_positive']
    gs2 = result['gs_negative']
    cell_types = result['cell_types']


    data=adata.raw.X.toarray()
    # Get cell-type by cell matrix
    scRNAseqData = pd.DataFrame(data, index=adata.raw.obs_names, columns=adata.raw.var_names)

    # Compute cell-type score fro each cell
    es_max = sctype_score(scRNAseqData=scRNAseqData, scaled=True, gs=gs, gs2=gs2, cell_types=cell_types)
    es_max.columns = cell_types
    es_max.index = scRNAseqData.index

    # Calculate neighborhood graph of cells (replace 'adata' with your actual AnnData object)
    sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_pca')
    # Perform clustering so that cell-type can be assigned to each cluster
    sc.tl.leiden(adata)
    # The cluster labels are stored in `adata.obs['leiden']`
    results = []
    for cl in adata.obs['leiden'].unique():
        cells_in_cluster = adata.obs_names[adata.obs['leiden'] == cl]
        es_max_cl = es_max.loc[cells_in_cluster].sum().sort_values(ascending=False)
        results.append(pd.DataFrame({
            'cluster': cl,
            'type': es_max_cl.index[:1],
            'scores': es_max_cl.values[:1],
            'ncells': len(cells_in_cluster)
            }))

    results = pd.concat(results)
    results.loc[results['scores'] < results['ncells'] / 4, 'type'] = 'Unknown'
    results.set_index('cluster', inplace=True)
    # Assign the cell type labels to the cells in the AnnData object
    adata.obs['cell_type'] = results.loc[adata.obs['leiden'], 'type'].values
    return adata




















