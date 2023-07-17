

for i=1:100

    Ix=genoMaps(i,:);
    Ixx=reshape(Ix,[rowN,colN]);
    
    figure(1)
    imagesc(Ixx)
    title(num2str(i))
    pause(2)
    
    
end
