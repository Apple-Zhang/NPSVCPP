clc;
clear;
close;

addpath utils
load flowers102_resnet34_fea.mat

Xtrain = double(tr_data);
Ytrain = double(tr_label(:));
Xtest  = double(te_data);
Ytest  = double(te_label(:));

nTrain = size(Xtrain, 1);

% construct graph
options.NeighborMode = 'KNN';
options.WeightMode   = 'HeatKernel';
options.k = floor(log(nTrain)/log(2));
G = constructW(Xtrain, options);

% casually set hyperparameters
param.d = 100;
param.G  = G;
param.kernel.kernelType = 'Gaussian';
param.c = 20;
param.r1 = 0.05;
param.r2 = 0.05;
param.gamma = 4;
param.mu = 0.01;

tic;
model  = NPSVCPP_train(Xtrain, Ytrain, param, "EPSILON", 1e-6);
result = NPSVCPP_predict(Xtest, Ytest, model);
toc;
disp(result.eval.accuracy);