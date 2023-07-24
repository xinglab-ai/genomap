# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:57:36 2023

@author: Md Tauhidul Islam
# This code is inspired by scType (https://github.com/IanevskiAleksandr/sc-type)
# We are in the process of using image matching techique for further enhancement
# of the cell annotation
"""
import pandas as pd
import numpy as np
import scipy
import openpyxl
from pybiomart import Dataset
import re

def sctype_score(scRNAseqData, scaled=True, gs=None, gs2=None, gene_names_to_uppercase=True, cell_types=None):
    # Ensure input is a pandas DataFrame
    if not isinstance(scRNAseqData, pd.DataFrame):
        print("scRNAseqData doesn't seem to be a DataFrame")
        return None
    
    # Check if DataFrame is empty
    if scRNAseqData.empty:
        print("The input scRNAseqData DataFrame is empty")
        return None

    # Marker sensitivity
    marker_stat = pd.Series([gene for sublist in gs for gene in sublist]).value_counts().sort_values(ascending=False)
    marker_sensitivity = pd.DataFrame({
        "score_marker_sensitivity": (len(gs)-marker_stat.values) / (len(gs) - 1),
        "gene_": marker_stat.index
    }) 

    # Convert gene names to uppercase
    if gene_names_to_uppercase:
        scRNAseqData.columns = scRNAseqData.columns.astype(str).str.upper()

    # Subselect genes found in data
    gs = [[gene for gene in sublist if gene in scRNAseqData.columns] for sublist in gs]
    gs2 = [[gene for gene in sublist if gene in scRNAseqData.columns] for sublist in gs2]

    cell_markers_genes_score = marker_sensitivity[marker_sensitivity['gene_'].isin(np.unique(np.concatenate(gs)))]

    Z = scRNAseqData.copy()
    
    #scanpy.pp.scale(Z)
    Z= scipy.stats.zscore(Z, axis=0, ddof=1)  
    Z = Z.fillna(0)

    # Multiply by marker sensitivity
    Z = Z.T
    for idx, row in cell_markers_genes_score.iterrows():
        Z.loc[row["gene_"]] *= row["score_marker_sensitivity"]
    Z = Z.T
    # Subselect only with marker genes
    Z = Z[np.unique(np.concatenate(gs + gs2))]    
    Z=Z.T
    score_series_list = []
    # Loop over each gene symbol (gss_) in gs
    
    for i in range(0,len(gs)):
        gss_=np.array(gs[i])
        gss2_=np.array(gs2[i])
        scores = []    
        # Loop over each column (j) in Z
        for j in range(Z.shape[1]):
            gs_z = Z.loc[gss_].iloc[:, j]  # Using iloc for integer-based indexing  
            gz_2 = Z.loc[gss2_].iloc[:, j] * -1  # Using iloc for integer-based indexing
            sum_t1 = np.sum(gs_z) / np.sqrt(len(gs_z))
            sum_t2 = np.sum(gz_2) / np.sqrt(len(gz_2))        
            if np.isnan(sum_t2):
                sum_t2 = 0        
            scores.append(sum_t1 + sum_t2)    
        # Convert the scores list to a pandas Series and add it to the score_series_list
        score_series_list.append(pd.Series(scores))
    # Concatenate the score series in score_series_list along axis 0 to create a DataFrame
    es = pd.concat(score_series_list, axis=1)

    # Remove rows with all NA or empty values
    es = es.dropna(how='all')
    es = es[~(es == "").all(axis=1)]

    return es

def gene_sets_prepare(path_to_db_file, cell_type):
    # Read the Excel file
    cell_markers = pd.read_excel(path_to_db_file, engine='openpyxl')

    # Select rows where 'tissueType' matches 'cell_type'
    cell_markers = cell_markers[cell_markers['tissueType'] == cell_type]
    # Apply the function to each row in the DataFrame 'cell_markers'
    # cell_markers['geneSymbolmore1'] = cell_markers['geneSymbolmore1'].apply(correct_gene_symbols)


    # Convert to string and clean 'geneSymbolmore1' and 'geneSymbolmore2' columns
    cell_markers['geneSymbolmore1'] = cell_markers['geneSymbolmore1'].astype(str).str.replace(" ", "")
    cell_markers['geneSymbolmore2'] = cell_markers['geneSymbolmore2'].astype(str).str.replace(" ", "")
    
 
    cell_markers["geneSymbolmore1"] = cell_markers["geneSymbolmore1"].apply(lambda x: correct_gene_symbols(x))
    cell_markers["geneSymbolmore2"] = cell_markers["geneSymbolmore2"].apply(lambda x: correct_gene_symbols(x))
    

    # Define a helper function to handle potential NaN values
    def process_genes(x):
        if x == 'nan':
            return []
        else:
            return sorted([i for i in x.split(",") if i not in ["NA", ""]])

    # Split 'geneSymbolmore1' and 'geneSymbolmore2' into lists of genes
    cell_markers['geneSymbolmore1'] = cell_markers['geneSymbolmore1'].apply(process_genes)
    cell_markers['geneSymbolmore2'] = cell_markers['geneSymbolmore2'].apply(process_genes)

    # Replace '///' with ',' and remove spaces
    cell_markers['geneSymbolmore1'] = cell_markers['geneSymbolmore1'].apply(lambda x: ",".join(x).replace("///", ",").replace(" ", ""))
    cell_markers['geneSymbolmore2'] = cell_markers['geneSymbolmore2'].apply(lambda x: ",".join(x).replace("///", ",").replace(" ", ""))

    # Split 'geneSymbolmore1' and 'geneSymbolmore2' into lists of genes again
    cell_markers['geneSymbolmore1'] = cell_markers['geneSymbolmore1'].str.split(",")
    cell_markers['geneSymbolmore2'] = cell_markers['geneSymbolmore2'].str.split(",")

    # Prepare 'gs' and 'gs2' lists
    gs = [genes for genes in cell_markers['geneSymbolmore1']]
    gs2 = [genes for genes in cell_markers['geneSymbolmore2']]
    
    # Prepare cell types list
    cell_types = list(cell_markers['cellName'])

    return {'gs_positive': gs, 'gs_negative': gs2, 'cell_types': cell_types}

def correct_gene_symbols(gene_symbols, species="human"):
    if isinstance(gene_symbols, str):
        gene_symbols = gene_symbols.split(",")
    gene_symbols = [gs.strip().replace(" ", "").upper() for gs in gene_symbols if gs.strip().upper() not in ["NA", ""]]
    gene_symbols = sorted(set(gene_symbols))

    if len(gene_symbols) > 0:
        # Assuming you have defined and imported the gene_symbol_mapping function
        # This function should perform the gene symbol mapping using pybiomart or any other method you have implemented.
        # gene_symbol_mapping(markers_all) should return a DataFrame with the "Suggested.Symbol" column.
        # For simplicity, I'll use a dummy function that returns an empty DataFrame here.
        #def gene_symbol_mapping(gene_symbols):
            #return pd.DataFrame(columns=["Suggested.Symbol"])

        suppressMessages = lambda x: x  # Suppressing messages in Python is not necessary

        markers_all = check_gene_symbols(gene_symbols)
        markers_all = markers_all.dropna(subset=["Suggested.Symbol"])
        markers_all = sorted(set(markers_all["Suggested.Symbol"]))

        return ",".join(markers_all)
    else:
        return ""


def check_gene_symbols(x, unmapped_as_na=True, map=None, species="human"):
    if species == "human":
        dataset_name = 'hsapiens_gene_ensembl'
    elif species == "mouse":
        dataset_name = 'mmusculus_gene_ensembl'
    else:
        raise ValueError("Species must be 'human' or 'mouse'")

    biomart = Dataset(name=dataset_name, host='http://www.ensembl.org')
    biomart_df = biomart.query(attributes=['external_gene_name', 'gene_biotype'])


    casecorrection = False
    if species == "human":
        casecorrection = True
        if map is None:
            #lastupdate = biomart_df["date"].max()
            #print("Maps last updated on:", lastupdate)
            map = biomart_df[['Gene name', 'Gene type']].drop_duplicates()
    elif species == "mouse" and map is None:
        #lastupdate = biomart_df["date"].max()
        #print("Maps last updated on:", lastupdate)
        map = biomart_df[['Gene name', 'Gene type']].drop_duplicates()
    else:
        if map is None:
            raise ValueError("If species is not 'human' or 'mouse', then map argument must be specified")

    if not isinstance(map, pd.DataFrame) or not set(map.columns) == {"Gene name", "Gene type"}:
        raise ValueError("If map is specified, it must be a dataframe with two columns named 'external_gene_name' and 'gene_biotype'")

    approved = [sym in map["Gene name"].values for sym in x]

    if casecorrection:
        x_casecorrected = [sym.upper() for sym in x]
        x_casecorrected = [re.sub(r"(.*C[0-9XY]+)ORF(.+)", r"\1orf\2", sym) for sym in x_casecorrected]
    else:
        x_casecorrected = x.split()

    approvedaftercasecorrection = [sym in map["Gene name"].values for sym in x_casecorrected]

    if x != " ".join(x_casecorrected):
        print("Human gene symbols should be all upper-case except for the 'orf' in open reading frames. The case of some letters was corrected.")

    alias = [sym in map["Gene name"].values for sym in x_casecorrected]

    suggested_symbols = []
    for i in range(len(x_casecorrected)):
        if approved[i]:
            suggested_symbols.append(x_casecorrected[i])
        else:
            if alias[i]:
                suggested_symbols.append(" /// ".join(map.loc[map["Gene name"] == x_casecorrected[i], "Gene name"]))
            elif approvedaftercasecorrection[i]:
                suggested_symbols.append(x_casecorrected[i])
            else:
                suggested_symbols.append(None)

    df = pd.DataFrame({"x": x, "Approved": approved, "Suggested.Symbol": suggested_symbols})
    df["Approved"][df["x"].isnull()] = False

    if not unmapped_as_na:
        df.loc[df["Suggested.Symbol"].isnull(), "Suggested.Symbol"] = df.loc[df["Suggested.Symbol"].isnull(), "x"]

    if df["Approved"].sum() != df.shape[0]:
        print("x contains non-approved gene symbols")

    return df


