close all;clear all; clc;
addpath ../common
addpath ../common/minFunc_2012/minFunc
addpath ../common/minFunc_2012/minFunc/compiled

% Load the MNIST data for this exercise.
% train.X and test.X will contain the training and testing images.
%   Each matrix has size [n,m] where:
%      m is the number of examples.
%      n is the number of pixels in each image.
% train.y and test.y will contain the corresponding labels (0 to 9).
binary_digits = false;
num_classes = 10;
[train,test] = ex1_load_mnist(binary_digits);

% Add row of 1s to the dataset to act as an intercept term.
train.X = [ones(1,size(train.X,2)); train.X]; 
test.X = [ones(1,size(test.X,2)); test.X];
train.y = train.y+1; % make labels 1-based.
test.y = test.y+1; % make labels 1-based.

% Training set info
m=size(train.X,2);
n=size(train.X,1);

% Train softmax classifier using minFunc
options = struct('MaxIter', 200);

% Initialize theta.  We use a matrix where each column corresponds to a class,
% and each row is a classifier coefficient for that class.
% Inside minFunc, theta will be stretched out into a long vector (theta(:)).
% We only use num_classes-1 columns, since the last column is always assumed 0.
theta = rand(n,num_classes-1)*0.001;
theta_mf = zeros(n, num_classes-1);

% Call minFunc with the softmax_regression_vec.m file as objective.
%
% TODO:  Implement batch softmax regression in the softmax_regression_vec.m
% file using a vectorized implementation.
%

%tic;
%theta_mf(:)=minFunc(@softmax_regression_vec, theta(:), options, train.X, train.y);
%fprintf('Optimization took %f seconds.\n', toc);
%theta_mf=[theta_mf, zeros(n,1)]; % expand theta to include the last class.

% Print out training accuracy.
%tic;
%accuracy = multi_classifier_accuracy(theta_mf,train.X,train.y);
%fprintf('Training accuracy: %2.1f%%\n', 100*accuracy);

% Print out test accuracy.
%accuracy = multi_classifier_accuracy(theta_mf,test.X,test.y);
%fprintf('Test accuracy: %2.1f%%\n', 100*accuracy);



lr = 0.00001;
tic;
theta_gd=softmax_regression_mine(theta, train.X, train.y, lr);
fprintf('Optimization took %f seconds.\n', toc);
theta_gd=[theta_gd, zeros(n,1)]; % expand theta to include the last class.

% Print out training accuracy.
tic;
accuracy = multi_classifier_accuracy(theta_gd,train.X,train.y);
fprintf('Training accuracy: %2.1f%%\n', 100*accuracy);

% Print out test accuracy.
accuracy = multi_classifier_accuracy(theta_gd,test.X,test.y);
fprintf('Test accuracy: %2.1f%%\n', 100*accuracy);



% % for learning curves
% global test
% global train
% test.err{end+1} = multi_classifier_accuracy(theta,test.X,test.y);
% train.err{end+1} = multi_classifier_accuracy(theta,train.X,train.y);
