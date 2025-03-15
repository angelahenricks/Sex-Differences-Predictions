% organize data
[data,~,files] = collateData('C:\Users\HenricksLab\Desktop\KHMatLabCode\BaseAnalysisFiles\Processed\',{'.mat'},{'pow','coh'},'avg','rel');
% Name features
nameVect = names({'ILL','CeAL','ILR','CeAR','NAcShL','NAcShR'},{'d','t','a','b','lg','hg'});
% Group Data FC = female control FV = female vapor MC = male control MV = male vapor
FC1 = cat(1,data{1}{22:24});
FC2 = cat(1,data{1}{35:37});
FC3 = cat(1,data{1}{60:62});
FC4 = cat(1,data{1}{69:71});
FC5 = cat(1,data{1}{81:86});
FC6 = cat(1,data{1}{88:90});
FC = [FC1; FC2; FC3; FC4; FC5; FC6];
FV1 = cat(1,data{1}{14});
FV2 = cat(1,data{1}{16:20});
FV3 = cat(1,data{1}{27:28});
FV4 = cat(1,data{1}{30:31});
FV5 = cat(1,data{1}{33:34});
FV6 = cat(1,data{1}{65:67});
FV7 = cat(1,data{1}{73:78});
FV = [FV1; FV2; FV3; FV4; FV5; FV6; FV7];
%
MC1 = cat(1,data{1}{1:3});
MC2 = cat(1,data{1}{9:11});
MC3 = cat(1,data{1}{40:42});
MC4 = cat(1,data{1}{51:53});
MC5 = cat(1,data{1}{91:96});
MC6 = cat(1,data{1}{111:113});
MC = [MC1; MC2; MC3; MC4; MC5; MC6];
MV1 = cat(1,data{1}{5:7});
MV2 = cat(1,data{1}{44:46});
MV3 = cat(1,data{1}{48:50});
MV4 = cat(1,data{1}{56:58});
MV5 = cat(1,data{1}{103:108});
MV = [MV1; MV2; MV3; MV4; MV5];
% create x and y values
Xf = [FC;FV];
Yf = [ones(21,1);zeros(21,1)];
Xm = [MC;MV];
Ym = [ones(21,1);zeros(18,1)];
%% Female models predicting male control vs. vapor
% Repeat 80:20 split, training, and testing model 100 times
for ii = 1:100
    % create indices (training on females; test on females; test on males)
    trainIndf = randperm(size(Yf,1),round(size(Yf,1)*.8));
    trainIndm = randperm(size(Ym,1),round(size(Ym,1)*.8));
    testIndf = logicFind(1,~ismember(1:size(Yf,1),trainIndf),'==');
    testIndm = logicFind(1,~ismember(1:size(Ym,1),trainIndm),'==');
    % create female training data
    trainXf = Xf(trainIndf,:);
    trainYf = Yf(trainIndf);
    % create male training data
    trainXm = Xm(trainIndm,:);
    trainYm = Ym(trainIndm);
    % create female testing data
    testXf = Xf(testIndf,:);
    testYf = Yf(testIndf);
    % create male testing data
    testXm = Xm(testIndm,:);
    testYm = Ym(testIndm);
    % Female-only model predict control vs. vapor in both M and F
    % Lasso real for females --> create model on female data
    cfg = lassoNetCfg({testXf,testYf},[],'n','y','y',100,'1se',[]);
    [~,Flambda,Fbeta,Ffits,Facc{ii},Fhist] = lassoNet(trainXf,trainYf,'binomial','deviance',1,5,1,cfg);
    % Lasso random for females --> create model on female data
    cfg = lassoNetCfg({testXf,testYf},[],'y','y','y',100,'1se',[]);
    [~,FlambdaR,FbetaR,FfitsR,FaccR{ii},FhistR] = lassoNet(trainXf,trainYf,'binomial','deviance',1,5,1,cfg);
    % Test female model on male data
    predY = cvglmnetPredict(Facc{ii}{1}.mdl{1},testXm,['lambda_',cfg.minTerm],'response');
    [FMacc{ii}{1}.x,FMacc{ii}{1}.y,~,FMacc{ii}{1}.auc] = perfcurve(testYm,predY,1);

    % Male-only model predict control vs. vapor in both M and F
    % Lasso real for males --> use Mfits to create male test
    cfg = lassoNetCfg({testXm,testYm},[],'n','y','y',100,'1se',[]);
    [~,Mlambda,Mbeta,Mfits,Macc{ii},Mhist] = lassoNet(trainXm,trainYm,'binomial','deviance',1,5,1,cfg);
    % Lasso random for males
    cfg = lassoNetCfg({testXm,testYm},[],'y','y','y',100,'1se',[]);
    [~,MlambdaR,MbetaR,MfitsR,MaccR{ii},MhistR] = lassoNet(trainXm,trainYm,'binomial','deviance',1,5,1,cfg);
    % Test male model on female data
    predY = cvglmnetPredict(Macc{ii}{1}.mdl{1},testXf,['lambda_',cfg.minTerm],'response');
    [MFacc{ii}{1}.x,MFacc{ii}{1}.y,~,MFacc{ii}{1}.auc] = perfcurve(testYf,predY,1);
end