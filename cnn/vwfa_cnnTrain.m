%% Convolution Neural Network Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started in building a single.
%  layer convolutional nerual network. In this exercise, you will only
%  need to modify cnnCost.m and cnnminFuncSGD.m. You will not need to 
%  modify this file.

%%======================================================================
%% STEP 0: Initialize Parameters and Load Data
%  Here we initialize some parameters used for the exercise.
clear all;close all;clc;
% Configuration



    
    
%% Load MNIST Train
%addpath ../common/;
%%addpath ../../programming/PA4/;

%path = '../../programming/PA4/images/';
%folders_all = dir(path);
%folders = {};
%for i=1:length(folders_all)
%    if (folders_all(i).name(1) ~= '.')
%        folders = [folders;folders_all(i).name];   
%    end
%end
%images = {};
%labels = [];
%curr_label = 1;
%for i=1:20
%    files_temp = dir( fullfile(path,folders{i},'*.png'));   
%    for j=1:length(files_temp)
%        images = [images;strcat(path, folders{i}, '/', files_temp(j).name)];
%        labels = [labels; curr_label];
%    end
%    curr_label = curr_label + 1;
%end

%I = randperm(length(labels));
%labels = labels(I);
%images = images(I);

%total_length = length(labels);
%images_validate = images(1:total_length/8);
%labels_validate = labels(1:total_length/8);
%images_test = images(1+total_length/8:total_length/4);
%labels_test = labels(1+total_length/8:total_length/4);
%images_train = images(total_length/4+1:end);
%labels_train = labels(total_length/4+1:end);

batch_size = 32;
imageDim = 112;
i=1;
%data_train = load_vwfa(images_train, imageDim);
%label_train = labels_train;

%save('train_data.mat', 'data_train');
%save('train_label.mat','label_train');
%data_test = load_vwfa(images_test, imageDim);
%label_test = labels_test;
%save('test_data.mat', 'data_test');
%save('test_label.mat','label_test');

%data_validate = load_vwfa(images_validate, imageDim);
%label_validate = labels_validate;
%save('validate_data.mat', 'data_validate');
%save('validate_label.mat','label_validate');

load train_data
load train_label
load validate_data
load validate_label
load test_data
load test_label

data_train = permute(data_train, [1 2 4 3]);
data_test = permute(data_test, [1 2 4 3]);
data_validate = permute(data_validate, [1 2 4 3]);

numClasses = max(label_train);  % Number of classes (MNIST images fall into 10 classes)
numChannels = 1;


%images = loadMNISTImages('../common/train-images-idx3-ubyte');
%images = reshape(images,imageDim,imageDim,1,[]);
%labels = loadMNISTLabels('../common/train-labels-idx1-ubyte');
%labels(labels==0) = 10; % Remap 0 to 10

DEBUG=false;  % set this to true to check gradient
if DEBUG
    % To speed up gradient checking, we will use a reduced network and
    % a debugging data set
    

    ei.lambda = 0.01;
    ei.layer_type = {'conv','nonlinear','pool', 'conv','nonlinear','pool','full', 'nonlinear', 'full'};
    
    ei.layer_param = cell(length(ei.layer_type),1);
    ei.layer_param{1}.filterDim = 5;
    ei.layer_param{1}.numFilters = 10;
    ei.layer_param{1}.numInput = 1;
    
    
    ei.layer_param{2}.activation_fun = 'relu';
    
    
    ei.layer_param{3}.pool_size = 2;
    %now local_size and pool_size must be the same for backprop
    ei.layer_param{3}.local_size = 2;
    ei.layer_param{3}.type = 'max';
    
    
    
    ei.layer_param{4}.filterDim = 5;
    ei.layer_param{4}.numFilters = 20;
    ei.layer_param{4}.numInput = 10;
    
    ei.layer_param{5}.activation_fun = 'relu';

    
    ei.layer_param{6}.pool_size = 2;
    %now local_size and pool_size must be the same for backprop
    ei.layer_param{6}.local_size = 2;
    ei.layer_param{6}.type = 'max';    
    
    ei.layer_param{7}.outputDim = 100;
    ei.layer_param{8}.activation_fun = 'relu';


    
    ei.layer_param{9}.outputDim = numClasses;
    
    

    
    db_images = data_train(:,:,:,1:10);
    db_labels = label_train(1:10);
    db_theta = mine_cnnInitParams(imageDim,numChannels,ei);
    
    [cost, grad_stack] = mine_cnnCost(ei, db_theta,db_images,db_labels,numClasses);
    grad = stackToParams(grad_stack);
    
    % Check gradients
    numGrad = mine_computeNumericalGradient( @(x) mine_cnnCost(ei,x,db_images,...
                                db_labels,numClasses), db_theta, 100);
 
    % Use this to visually compare the gradients side by side
    disp([numGrad grad]);
    
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    % Should be small. In our implementation, these values are usually 
    % less than 1e-9.
    disp(diff); 
 
    assert(diff < 1e-8,...
        'Difference too large. Check your gradient computation again');
    
end;


%%======================================================================
%% STEP 3: Learn Parameters
%  Implement minFuncSGD.m, then train the model.


    ei.lambda = 0.00;
    ei.layer_type = {'conv','nonlinear','pool', 'conv','nonlinear','pool','full', 'nonlinear', 'full'};
    
    ei.layer_param = cell(length(ei.layer_type),1);
    ei.layer_param{1}.filterDim = 5;
    ei.layer_param{1}.numFilters = 10;
    ei.layer_param{1}.numInput = 1;
    
    
    ei.layer_param{2}.activation_fun = 'relu';
    
    
    ei.layer_param{3}.pool_size = 2;
    %now local_size and pool_size must be the same for backprop
    ei.layer_param{3}.local_size = 2;
    ei.layer_param{3}.type = 'max';
    
    
    
    ei.layer_param{4}.filterDim = 5;
    ei.layer_param{4}.numFilters = 20;
    ei.layer_param{4}.numInput = 10;
    
    ei.layer_param{5}.activation_fun = 'relu';

    
    ei.layer_param{6}.pool_size = 2;
    %now local_size and pool_size must be the same for backprop
    ei.layer_param{6}.local_size = 2;
    ei.layer_param{6}.type = 'max';    
    
    ei.layer_param{7}.outputDim = 100;
    ei.layer_param{8}.activation_fun = 'relu';


    
    ei.layer_param{9}.outputDim = numClasses;
    
    

    
theta = mine_cnnInitParams(imageDim,numChannels, ei);
options.epochs = 2;
options.minibatch = 32;
options.alpha = 1e-1;
options.momentum = .95;

opttheta = mine_minFuncSGD(@(x,y,z) mine_cnnCost(ei, x,y,z,numClasses),theta,data_train,label_train,options);

%%======================================================================
%% STEP 4: Test
%  Test the performance of the trained model using the MNIST test set. Your
%  accuracy should be above 97% after 3 epochs of training

%testImages = loadMNISTImages('../common/t10k-images-idx3-ubyte');
%testImages = reshape(testImages,imageDim,imageDim,1, []);

%testLabels = loadMNISTLabels('../common/t10k-labels-idx1-ubyte');
%testLabels(testLabels==0) = 10; % Remap 0 to 10

[~,cost,preds]=mine_cnnCost(ei, opttheta,data_test,label_test,numClasses,...
                true);

acc = sum(preds==testLabels)/length(preds);

% Accuracy should be around 97.4% after 3 epochs
fprintf('Accuracy is %f\n',acc);
