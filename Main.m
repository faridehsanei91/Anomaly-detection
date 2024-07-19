clc;
clear;
close all;

data=xlsread('prj1');

target=data(1:130,1);
%target=[];
%% make filters
L=[1 4 6 4 1];
E=[-1 -2 0 2 1];
S=[-1 0 2 0 -1];
R=[1 -4 6 -4 1];

M{1}=((L'*E)+(E'*L))/2;
M{2}=((L'*S)+(S'*L))/2;
M{3}=((L'*R)+(R'*L))/2;
M{4}=((E'*S)+(S'*E))/2;
M{5}=((E'*R)+(R'*E))/2;
M{6}=((R'*S)+(S'*R))/2;
M{7}=S'*S;
M{8}=E'*E;
M{9}=R'*R;

Data=[];
%t=[];
%%
%read data & image cropping
for i=1:130
    im=imread(['all-mias\2 ('  num2str(i) ').pgm']);
%     imshow(im)
%     figure
    im=im(1024-data(i,4)-64:1024-data(i,4)+63,data(i,3)-64:data(i,3)+63);
    %apply filters
    m={};
    for j=1:1:9
        m{j}=imresize(imfilter(im,M{j},'conv'),[120 120]);
    end
%     imshow(m{1});
%     break
    %%
    %feature extraction
    
%         M=15;
%         N=15;
        [f1_me , f1_v]=calculate(m{1});
        [f2_me , f2_v]=calculate(m{2});
        [f3_me , f3_v]=calculate(m{3});
        [f4_me , f4_v]=calculate(m{4});
        [f5_me , f5_v]=calculate(m{5});
        [f6_me , f6_v]=calculate(m{6});
        [f7_me , f7_v]=calculate(m{7});
        [f8_me , f8_v]=calculate(m{8});
        [f9_me , f9_v]=calculate(m{9});
%         f1=colfilt(m{1},[M N],'distinct',@mean);
%         f2=colfilt(m{2},[M N],'distinct',@mean);
%         f3=colfilt(m{3},[M N],'distinct',@mean);
%         f4=colfilt(m{4},[M N],'distinct',@mean);
%         f5=colfilt(m{5},[M N],'distinct',@mean);
%         f6=colfilt(m{6},[M N],'distinct',@mean);
%         f7=colfilt(m{7},[M N],'distinct',@mean);
%         f8=colfilt(m{8},[M N],'distinct',@mean);
%         f9=colfilt(m{9},[M N],'distinct',@mean);
    
    %%
    %     MyFun=@(z) (sum(z(:))-mean(z))/M*N;
%     
%        f10=colfilt(double(m{1}),[M N],'distinct',@var);
    %     f11=colfilt(m2,[M N],'sliding',MyFun);
    %     f12=colfilt(m3,[M N],'sliding',MyFun);
    %     f13=colfilt(m4,[M N],'sliding',MyFun);
    %     f14=colfilt(m5,[M N],'sliding',MyFun);
    %     f15=colfilt(m6,[M N],'sliding',MyFun);
    %     f16=colfilt(m7,[M N],'sliding',MyFun);
    %     f17=colfilt(m8,[M N],'sliding',MyFun);
    %     f18=colfilt(m9,[M N],'sliding',MyFun);
    %
    

    
    Data=[Data;f1_me f2_me f3_me f4_me f5_me f6_me f7_me f8_me f9_me f1_v f2_v f3_v f4_v f5_v f6_v f7_v f8_v f9_v];
    
end



%classification
X=floor(0.7*size(Data,1));

inptrain=Data(1:X,:)';
inptest=Data(X+1:end,:)';
targettrain=target(1:X,:)';
targettest=target(X+1:end,:)';

%% Problem Definition



nVar=1152;            % Number of Decision Variables

VarSize=[1 nVar];   % Decision Variables Matrix Size


%% GA Parameters

MaxIt=100;      % Maximum Number of Iterations

nPop=10;        % Population Size

pc=0.8;                 % Crossover Percentage
nc=2*round(pc*nPop/2);  % Number of Offsprings (Parnets)

pm=0.3;                 % Mutation Percentage
nm=round(pm*nPop);      % Number of Mutants

mu=0.02;         % Mutation Rate

%% Initialization



empty_individual.Position=[];
empty_individual.Cost=[];

pop=repmat(empty_individual,nPop,1);

for i=1:nPop
    
    % Initialize Position
    pop(i).Position=randi([0 1],VarSize);
    
    % Evaluation
    pop(i).Cost=MinOne(pop(i).Position,inptrain,targettrain,inptest,targettest);
    
end

% Sort Population
Costs=[pop.Cost];
[Costs, SortOrder]=sort(Costs);
pop=pop(SortOrder);

% Store Best Solution
BestSol=pop(1);

% Array to Hold Best Cost Values
BestCost=zeros(MaxIt,1);


%% Main Loop

for it=1:MaxIt
        
    
    % Crossover
    popc=repmat(empty_individual,nc/2,2);
    for k=1:nc/2
              
        
            i1=randi([1 nPop]);
            i2=randi([1 nPop]);
                  
        % Select Parents
        p1=pop(i1);
        p2=pop(i2);
        
        % Apply Crossover
        [popc(k,1).Position popc(k,2).Position]=Crossover(p1.Position,p2.Position);
        
        % Evaluate Offsprings
        popc(k,1).Cost=MinOne(popc(k,1).Position,inptrain,targettrain,inptest,targettest);
        popc(k,2).Cost=MinOne(popc(k,2).Position,inptrain,targettrain,inptest,targettest);
        
    end
    popc=popc(:);
    
    % Mutation
    popm=repmat(empty_individual,nm,1);
    for k=1:nm
        
        % Select Parent
        i=randi([1 nPop]);
        p=pop(i);
        
        % Apply Mutation
        popm(k).Position=Mutate(p.Position,mu);
        
        % Evaluate Mutant
        popm(k).Cost=MinOne(popm(k).Position,inptrain,targettrain,inptest,targettest);
        
    end
    
    % Create Merged Population
    pop=[pop
         popc
         popm];
     
    % Sort Population
    Costs=[pop.Cost];
    [Costs, SortOrder]=sort(Costs);
    pop=pop(SortOrder);
    
   
    % Truncation
    pop=pop(1:nPop);
    Costs=Costs(1:nPop);
    
    % Store Best Solution Ever Found
    BestSol=pop(1);
    
    % Store Best Cost Ever Found
    BestCost(it)=BestSol.Cost;
    
   
    
    % Show Iteration Information
    disp(['Iteration : ' num2str(it) '     Best Cost =  ' num2str(BestCost(it))]);
    
end

