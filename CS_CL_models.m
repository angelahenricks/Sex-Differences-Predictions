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

% create corticostriatal (CS) and corticolimbic (CL) list of variables
CS = [1:6, 13:18, 25:36, 43:48, 55:66, 97:108, 121:126];
CL = [1:24, 37:54, 67:78, 91:96];

% Female models predicting control vs. vapor with CS only
% Lasso real for female (F) corticostriatal (CS)
    cfg = lassoNetCfg([],[],'n','y','y',100,'1se',[]);
     [~,Flambda_CS,Fbeta_CS,Ffits_CS,Facc_CS,Fhist_CS] = lassoNet(Xf(:,CS),Yf,'binomial','auc',1,5,1,cfg);
% Lasso random for female (F) corticostriatal (CS)
    cfg = lassoNetCfg([],[],'y','y','y',100,'1se',[]);
    [~,FlambdaR_CS,FbetaR_CS,FfitsR_CS,FaccR_CS,FhistR_CS] = lassoNet(Xf(:,CS),Yf,'binomial','auc',1,5,1,cfg);

% Male Models predicting control vs. vapor with CS only
% Lasso real for CS
    cfg = lassoNetCfg([],[],'n','y','y',100,'1se',[]);
    [~,Mlambda_CS,Mbeta_CS,Mfits_CS,Macc_CS,Mhist_CS] = lassoNet(Xm(:,CS),Ym,'binomial','auc',1,5,1,cfg);
    
% Lasso random for male (M) corticostriatal (CS)
    cfg = lassoNetCfg([],[],'y','y','y',100,'1se',[]);
    [~,MlambdaR_CS,MbetaR_CS,MfitsR_CS,MaccR_CS,MhistR_CS] = lassoNet(Xm(:,CS),Ym,'binomial','auc',1,5,1,cfg);

% Female models predicting control vs. vapor with CL only
% Lasso real for female (F) corticolimbic (CL)
    cfg = lassoNetCfg([],[],'n','y','y',100,'1se',[]);
     [~,Flambda_CL,Fbeta_CL,Ffits_CL,Facc_CL,Fhist_CL] = lassoNet(Xf(:,CL),Yf,'binomial','auc',1,5,1,cfg);
% Lasso random for female (F) corticolimbic (CL)
    cfg = lassoNetCfg([],[],'y','y','y',100,'1se',[]);
    [~,FlambdaR_CL,FbetaR_CL,FfitsR_CL,FaccR_CL,FhistR_CL] = lassoNet(Xf(:,CL),Yf,'binomial','auc',1,5,1,cfg);

% Male Models predicting control vs. vapor with CL only
% Lasso real for male (M) corticolimbic (CL)
    cfg = lassoNetCfg([],[],'n','y','y',100,'1se',[]);
    [~,Mlambda_CL,Mbeta_CL,Mfits_CL,Macc_CL,Mhist_CL] = lassoNet(Xm(:,CL),Ym,'binomial','auc',1,5,1,cfg);
% Lasso random for male (M) corticolimbic (CL)
    cfg = lassoNetCfg([],[],'y','y','y',100,'1se',[]);
    [~,MlambdaR_CL,MbetaR_CL,MfitsR_CL,MaccR_CL,MhistR_CL] = lassoNet(Xm(:,CL),Ym,'binomial','auc',1,5,1,cfg);