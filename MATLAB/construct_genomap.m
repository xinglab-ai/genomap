function genomaps=construct_genomap(data,rowNum,colNum)
%   construct_genomap 
%   genomaps=construct_genomap(data,rowNum,colNum)
%    constructs genomaps for a tabulated gene expression data
%
%   Inputs:
%   data: double matrix, high dimensional gene expression data 
%   in tabular format, i.e., rows denote cells and columns denote the 
%   genes. 
%   rowNum: integer, number of rows in a genomap
%   colNum: integer, number of columns in a genomap
%   Outputs:
%   genomaps: constructed genomaps
%   
%   Written by Md Tauhidul Islam, Ph.D., Postdoc, Radiation Oncology,
%   Stanford University, tauhid@stanford.edu
%%
epsilonMY=0;
[~,geneNum]=size(data);
if geneNum>rowNum*colNum
    [data,~]=selectVarGenes(data,rowNum*colNum);
end
[cellNum,geneNum]=size(data);
data=zscore(data);

% create interaction matrix
interactMattoDIS=createInteractionMatrix(data);

% create mesh distance matrix
gridDistMat=createMeshDistanceMatrix(rowNum,colNum);


if (geneNum<rowNum*colNum)    
  gridDistMat=gridDistMat(1:geneNum,1:geneNum);  
end


Mu=ones(1,geneNum)/(geneNum);

% Other parameter settings in optimal transport
setparams();
options.log_domain = 0; % use stabilization or not


[gamma,~,~,~] = perform_gw_sinkhorn(interactMattoDIS,gridDistMat,Mu',Mu',epsilonMY, options);

projMat=gamma*geneNum;
projectionM=data*projMat;

genomaps=zeros(rowNum,colNum,1,cellNum);
for i=1:cellNum
    
    dx = projectionM(i, :);    
    fullVec = zeros(1,rowNum*colNum);    
    fullVec(1,1:geneNum) = dx;
    ex = reshape(fullVec, [rowNum, colNum]);
    genomaps(:,:,1,i)=ex;

end
