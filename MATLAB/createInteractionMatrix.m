

function interactionMatrix=createInteractionMatrix(data)
%   createInteractionMatrix 
%   interactionMatrix=createInteractionMatrix(data)
%    computes the paiwise interaction matrix among the genes
%
%   Inputs:
%   data: double matrix, high dimensional gene expression data 
%   in tabular format, i.e., rows denote cells and columns denote the 
%   genes. 
%   Outputs:
%   interactionMatrix: paiwise interaction matrix
%   
%   Written by Md Tauhidul Islam, Ph.D., Postdoc, Radiation Oncology,
%   Stanford University, tauhid@stanford.edu
%%


interactionMatrix=pdist2(data',data','correlation');
