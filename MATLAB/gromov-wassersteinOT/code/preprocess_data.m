load reducedData
load groundTruth


Xg=X(1:484,:);

Dg=pdist2(Xg,Xg);


rowN=22;
colN=22;


for i=-rowN/2:rowN/2-1
    for j=-colN/2:colN/2-1
        meshV(i+rowN/2+1,j+colN/2+1)=sqrt((i)^2+(j)^2);
    end
end

meshX=meshV(:);

Dm=pdist2(meshX,meshX);

Mu=ones(1,rowN*rowN)/(rowN*rowN);
epsiL=0.5;