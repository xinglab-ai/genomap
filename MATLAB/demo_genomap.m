
%% Demonstration of Genomap construction from Tabula Muris scRNA-seq dataset
% (schaum et al., 2018, Nature)
% @author: Md Tauhidul Islam, Physical Science Research Scientist,
% Department of Radiation Oncology, Stanford University
%%


clear
% Add necessary code folders
addpath(genpath('gromov-wassersteinOT'))

% Load data
load TMdata
data=double(data);
% Row and column number of Genomap images
rowN=33;
colN=33;

genomaps=construct_genomap(data,rowN,colN);



%% Visualize the genomaps

% Load data label for visualization
load label_TMData
load index_TM

GT=grp2idx(label_TMData);
GTX=categorical(GT);
dataMat_CNNtrain=genomaps(:,:,:,indxTrain);
dataMat_CNNtest=genomaps(:,:,:,indxTest);
groundTruthTest=GTX(indxTest);
groundTruthTrain=GTX(indxTrain);

label_TMDataU=unique(label_TMData);

[GTsorted,idxGT]=sort(groundTruthTrain);

dataSorted=dataMat_CNNtrain(:,:,:,idxGT);

[unique_value,idxOcc]=unique(GTsorted);

numIm=[1:100];

idxX=[31 8 11 28 17 52 26 3 37 29];

for ii=1:length(idxOcc)
    
    if (sum(ii==idxX)>0)
        
        for jj=1:10
       
            
            im=squeeze(dataSorted(:,:,1,idxOcc(ii)+jj));
            
            figure(1)
            imagesc(im)
            title(strcat('cell class\_',string(label_TMDataU(ii))))
            axis off

            pause(2)
        end
    else
    end
end


