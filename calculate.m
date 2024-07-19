function [me , v]=calculate(im)
me=[];
v=[];
for i=1:15:120
    for j=1:15:120
        imtemp=im(i:i+14,j:j+14);
        me=[me mean(mean(imtemp))];
        v=[v var(var(double(imtemp)))];
    end
end
end