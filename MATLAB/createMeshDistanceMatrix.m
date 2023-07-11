
function meshDistanceMatrix=createMeshDistanceMatrix(rowN,colN)
%   createMeshDistanceMatrix 
%   meshDistanceMatrix=createMeshDistanceMatrix(rowN,colN)
%    computes the distance matrix in Euclidean space
%
%   Inputs:
%   rowNum: integer, number of rows in a genomap
%   colNum: integer, number of columns in a genomap
%   genes. 
%   Outputs:
%   meshDistanceMatrix: paiwise mesh distance matrix
%   
%   Written by Md Tauhidul Islam, Ph.D., Postdoc, Radiation Oncology,
%   Stanford University, tauhid@stanford.edu
%%

if (even_or_odd(rowN))
    r=1;
else
    rowN=rowN-1;
    r=0;
end
if (even_or_odd(colN))
    c=1;
else
    colN=colN-1;
    c=0;
end

if (r==1)&&(c==1)
for i=-rowN/2:rowN/2-1
    for j=-colN/2:colN/2-1
        meshV(i+rowN/2+1,j+colN/2+1)=sqrt((i)^2+(j)^2);
    end
end
elseif (r==0)&&(c==1)
for i=-rowN/2:rowN/2
    for j=-colN/2:colN/2-1
        meshV(i+rowN/2+1,j+colN/2+1)=sqrt((i)^2+(j)^2);
    end
end
elseif (r==1)&&(c==0)
for i=-rowN/2:rowN/2-1
    for j=-colN/2:colN/2
        meshV(i+rowN/2+1,j+colN/2+1)=sqrt((i)^2+(j)^2);
    end
end
elseif (r==0)&&(c==0)
for i=-rowN/2:rowN/2
    for j=-colN/2:colN/2
        meshV(i+rowN/2+1,j+colN/2+1)=sqrt((i)^2+(j)^2);
    end
end
end
meshX=meshV(:);

meshDistanceMatrix=pdist2(meshX,meshX);