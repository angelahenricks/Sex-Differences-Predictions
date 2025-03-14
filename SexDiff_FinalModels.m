% organize data
[data,~,files] = collateData('/home/kelly.hewitt/processedfiles/',{'.mat'},{'pow','coh'},'avg','rel');
% Name features
nameVect = names({'ILL','CeAL','ILR','CeAR','NAcShL','NAcShR'},{'d','t','a','b','lg','hg'});
% Group Data FC = female control FV = female vapor MC = male control MV = male vapor
FC1 = cat(1,data{1}{22:26});
FC2 = cat(1,data{1}{35:39});
FC3 = cat(1,data{1}{60:64});
FC4 = cat(1,data{1}{69:71});
FC5 = cat(1,data{1}{80:85});
FC6 = cat(1,data{1}{87:89});
FC = [FC1; FC2; FC3; FC4; FC5; FC6];

% Label IDs to FC variable
FCid = [129,129,129,129,129,133,133,133,133,133,140,140,140,140,140,143,143,143,66,66,66,67,67,67,70,70,70];

% What are all the unique IDs? 
FCuniq = unique(FCid);

% grab one random animal for testing
testFC = FCuniq(randperm(numel(FCuniq),1));

% grab rest of the animals for training
trainFC = FCuniq(~ismember(FCuniq,testFC));

% find the indecise that correspond with the test ID just identified
FCtestInd = find(FCid==testFC);

% find the indices that correspond with the TRAIN IDs just identified
FCtrainInd = find(FCid~=testFC);

FV1 = cat(1,data{1}{14:21});
FV2 = cat(1,data{1}{27:34});
FV3 = cat(1,data{1}{65:68});
FV4 = cat(1,data{1}{72:79});
FV5 = cat(1,data{1}{86});
FV = [FV1; FV2; FV3; FV4; FV5];

% Label IDs for FV variable
FVid = [127,127,127,127,128,128,128,128,130,130,130,130,132,132,132,132,142,142,142,142,61,61,61,63,63,63,65,65,69]

% What are all the unique IDs? 
FVuniq = unique(FVid);

% Once it reaches this line, this is the repeat spot
for ii = 1:100

% grab one random animal for testing
testFV = FVuniq(randperm(numel(FVuniq),1));

% grab rest of the animals for training
trainFV = FVuniq(~ismember(FVuniq,testFV));

% find the indices that correspond with the TEST ID just identified
FVtestInd = find(FVid==testFV);

% find the indices that correspond with the TRAIN IDs just identified
FVtrainInd = find(FVid~=testFV);

% create x and y values
trainXf = [FC(FCtrainInd,:);FV(FVtrainInd,:)];
trainYf = [ones(numel(FCtrainInd),1);zeros(numel(FVtrainInd),1)];
testXf = [FC(FCtestInd,:);FV(FVtestInd,:)];
testYf = [ones(numel(FCtestInd),1);zeros(numel(FVtestInd),1)];

% Female models predicting control vs. vapor
% Lasso real
    cfg = lassoNetCfg({testXf,testYf},[],'n','y','y',1,'1se',[]);
    [~,Flambda{ii},Fbeta{ii},Ffits{ii},Facc{ii},Fhist{ii}] = lassoNet(trainXf,trainYf,'binomial','auc',1,5,1,cfg);
% between the ii and the 'end' repeats! Once it hits 100, it will move on
end
% KH did stuff after this - high probability for error
for jj = 1:126
        mdl = fitglm(testXf(:,jj),testYf,'distribution','binomial','binomialsize',30);
        pred = predict(mdl,testXf(:,jj));
        [~,~,~,Fauc(:,jj)] = perfcurve(testYf,pred,1);
end