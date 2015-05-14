% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

%% setup environment
% experiment information
% a struct containing network layer sizes etc
clear all;close all;clc;
ei = [];

% add common directory to your path for
% minfunc and mnist data helpers
addpath ../common;
addpath(genpath('../common/minFunc_2012/minFunc'));

%% load mnist data
[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();


%% load expression data
%load PreprocessedData_expression_600          % 48%  sgd 47.86%
%load GaborData_expression

%%load jaffe expression data
%load jaffe_PreprocessedData_expression_600
%load jaffe_GaborData_expression


%load POFA_PreprocessedData_expression_30  %this is better 84%   sgd 76% whith 40 hidden units
%load POFA_GaborData_expression  %this is not so good <80%         sgd 80% with 100 hidden unit

%load PF_POFA_GaborData_expression    % 70%                 sgd 60% with 100 hidden units
%load PF_POFA_PreprocessedData_expression_30  % this is good 87.5%  sgd 87.5% with 40 hidden units


%% load identity data
%load GaborData_mine
%load PreprocessedData_mine      %96.5%           sgd 96% with 20 hidden units


%data_train = data.train;
%data_test = data.test;
%labels_train = data.train_label;
%labels_test = data.test_label;


 s=std(data_train,[],2);
 m=mean(data_train,2);
 data_train=bsxfun(@minus, data_train, m);
 data_train=bsxfun(@rdivide, data_train, s+.1);
 data_test=bsxfun(@minus, data_test, m);
 data_test=bsxfun(@rdivide, data_test, s+.1);
  
I = randperm(size(data_train,2));
data_train = data_train(:,I);
labels_train = labels_train(I);

I = randperm(size(data_test,2));
data_test = data_test(:,I);
labels_test = labels_test(I);


%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% the architecture specified below should produce  100% training accuracy
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)

% dimension of input features
ei.input_dim = size(data_train,1);
% number of output classes
ei.output_dim = max(labels_train);
% sizes of all hidden layers and the output layer
ei.layer_sizes = [256, ei.output_dim];
% scaling parameter for l2 weight regularization penalty
ei.lambda = 0.01;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
ei.activation_fun = 'tanh';
ei.momentum = 0.9;

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% setup minfunc options
options = [];
options.display = 'iter';
options.maxFunEvals = 1e3;
options.Method = 'lbfgs';

%% run training
%[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost_regularized,...
%    params,options,ei, data_train, labels_train);
lr = 0.01;
opt_params = mini_sgd_dnn(params, ei, data_train, labels_train,lr,100,data_test, labels_test);
    
%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost_regularized( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f\n', acc_test);

[~, ~, pred] = supervised_dnn_cost_regularized( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f\n', acc_train);
