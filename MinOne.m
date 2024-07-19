function cost=MinOne(X,inptrain,targettrain,inptest,targettest)
index=find(X==1);
inptrain=inptrain(index,:);
inptest=inptest(index,:);

net=newff(inptrain,targettrain,120,{'tansig','tansig'},'traingdx');

net=train(net,inptrain,targettrain);
out=sim(net,inptest);
out(out>=0)=1;
out(out<0)=-1;
correction=size(find((out-targettest)==0),2);
Accuracy=(correction/size(targettest,2))*100;

 
 cost=100-Accuracy;
end