function errorTest = fitELMKernel(Xtrain, Ytrain, XTest, YTest, S, C)

% ELM Kernel Steps:
% 1) Compute Kernel matrix
% 2) Calculate Omega
% 3) Estimate output

[N, K] = size(Xtrain);
[Ntest, Ktest] = size(XTest);

%No se calcula w si no la omega y P
Omega_train = kernel_matrix(Xtrain,'RBF_kernel',S);

%Calculamos el kernel Matrix de h(x) de todo con H que es todo tambn
KernelMatrix = kernel_matrix(Xtrain,'RBF_kernel',S,XTest);

% Compute Beta
BetaP = ((Omega_train+(speye(N)/C))\(Ytrain));

% Compute output
Output = KernelMatrix'*BetaP;

[~, indexPredicted] = max(Output');
[maximunValues, indexReal] = max(YTest');
errorTest = sum(indexPredicted' == indexReal')./size(XTest,1);
errorTest=1-errorTest;



%omega=rand(S,k);
%Htrain=calculate(Xtrain)
%beta=(Htrain'*Htrain + (1/C))*Htrain'*YTrain;
%Htest=calculate(Xtest);
%output=Htest*beta;
%error=abs(output-Ytest)^2;