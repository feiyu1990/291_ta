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
%[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();

%load PreprocessedData_new
load PreprocessedData_expression

data_train = data.train;
data_test = data.test;
labels_train = data.train_label;
labels_test = data.test_label;

I = randperm(size(data_train,2));
data_train = data_train(:,I);
labels_train = labels_train(I);

%I = randperm(size(data_test,2));
%data_test = data_test(:,I);
%labels_test = labels_test(I);


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
ei.layer_sizes = [20, ei.output_dim];
% scaling parameter for l2 weight regularization penalty
ei.lambda = 0;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
ei.activation_fun = 'logistic';

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% setup minfunc options
options = [];
options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';

%% run training
%[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
%    params,options,ei, data_train, labels_train);
lr = 0.1;
opt_params = sgd_dnn(params, ei, data_train, labels_train,lr,5000, data_test, labels_test);
    
%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f\n', acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f\n', acc_train);
