clear all;

rng(1);

% Load the dataset
D = load('handwriting.mat');

%displayData(D.X(490:510,:));

X = D.X;

% Number of number of patterns, attributes and classes
[N, K] = size(X);
J = 10;

% Number of hidden nodes
S = optimizableVariable('s', [10^(-3),10^(3)], 'Type', 'real');       %%Nuestra Sigma

% Regularization parameter
C = optimizableVariable('c', [10^(-3),10^(3)], 'Type', 'real');

Y = zeros(N,J);

% Generate the class label
for i=1:J
    Y(1+(500*(i-1)):500*i,i) = 1;
end

% Scaling the data
Xscaled = (X - min(X))./(max(X)-min(X));

% Remove NaN columns
Xscaled = Xscaled(:,any(~isnan(Xscaled)));

% Recompute the new problem dimension
[N, K] = size(Xscaled);



% Calculate real values
[maximunValues, indexReal] = max(Y');



%%Hasta aqui era ELM sin cross validacion

% Hold-out cross-validation
CVHO = cvpartition(indexReal','HoldOut',0.25);

% Data partitioning
XscaledTrain = Xscaled(CVHO.training(1),:);
XscaledTest = Xscaled(CVHO.test(1),:);

[NTrain, KTrain]=size(XscaledTrain);

YTrain = Y(CVHO.training(1),:);
YTest = Y(CVHO.test(1),:);

[~, indexTrain] = max(YTrain');

CVHOV = cvpartition(indexTrain','HoldOut',0.25);

XscaledTrainVal = XscaledTrain(CVHOV.training(1),:);
XscaledVal = XscaledTrain(CVHOV.test(1),:);

[NtrainVal, KtrainVal]=size(XscaledTrainVal);

YTrainVal = YTrain(CVHOV.training(1),:);
YVal = YTrain(CVHOV.test(1),:);

%Procedemos a crear la funcion Fitness 

fun=@(X)fitELMKernel(XscaledTrainVal, YTrainVal, XscaledVal, YVal, X.s, X.c);
%Ahora hacemos la Optimizacion Bayesiana para hacer parejas de
%hiperparametros S y C y buscar la optima

results=bayesopt(fun, [S,C], 'Verbose',0,...
    'AcquisitionFunctionName','expected-improvement-plus');

A = table2array(results.XAtMinEstimatedObjective);

%Variables Optima
Sopt=A(1);
Copt=A(2);

resultsTest=fitELMKernel(XscaledTrain, YTrain, XscaledTest, YTest, Sopt, Copt);

%Reportar CCR
CCRKernel=1-resultsTest



