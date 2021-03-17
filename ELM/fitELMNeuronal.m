function errorTest = fitELMNeuronal(Xtrain, Ytrain, XTest, YTest, S, C)

[N, K] = size(Xtrain);
[Ntest, Ktest] = size(XTest);

% W within -1 and 1
W = rand(S,K)*2-1;
Bias = rand(S,1);

BiasMatrix = repmat(Bias,1,N);

% P dimension NxS
P = (W*Xtrain'+ BiasMatrix)';

% H dimension NxS
H = 1./(1+exp(-P));

% Compute Beta
Beta = inv(H'*H + (eye(S)./C))*H'*Ytrain;

BiasMatrixtest = repmat(Bias,1,Ntest);

% Ptest dimension NxS
Ptest = (W*XTest'+ BiasMatrixtest)';

% Htest dimension NxS
Htest = 1./(1+exp(-Ptest));

% Compute outputTest
OutputTest = Htest*Beta;

%error=abs(OutputTest-YTest);
%errrorTest=error.^2;

[~, indexPredicted] = max(OutputTest');
[maximunValues, indexReal] = max(YTest');
errorTest = sum(indexPredicted' == indexReal')./size(XTest,1);
errorTest=1-errorTest;

%Al final siempre reportamos el ccr que es 1-error.