%Preprocessed_30 ~72% for mlp & softmax regression


clear all;close all;clc;
ei = [];

% add common directory to your path for
% minfunc and mnist data helpers
addpath ../common;
addpath(genpath('../common/minFunc_2012/minFunc'));


root = strcat(pwd, '/../../programming/PA3/');
files_temp = dir( fullfile(root,'*.pgm'));   
files = {};
for i=1:length(files_temp)
    if (files_temp(i).name(1) ~= '.')
        files = [files;files_temp(i).name];   
    end
end

label = zeros(length(files),1);
identity = {};
for i=1:length(files)
    identity = [identity; files{i}(1:2)];
end
identity_unique = unique(identity);

%for i=1:length(identity_unique)
%    disp(i)
%    Preprocess_pofa_exp_new( identity_unique(i),30 );
%end
acc = zeros(length(identity_unique),1);
count = zeros(length(identity_unique),1);

%num_pca = 2560;
%for i=2:2
for i=1:length(identity_unique)
    name = strcat(identity_unique(i), '_zscore_POFA_PreprocessedData_expression_30.mat');
    %name = strcat(identity_unique(i), '_zscore_POFA_GaborData_expression.mat');
    load (name{1})  
    data_train = data.train;
    data_test = data.test;
    labels_train = data.train_label;
    labels_test = data.test_label;

    
    %scale_all=data_train;
    %scale_all_test=data_test;
    %mean_images=mean(scale_all,2);            
    %mean_subst=scale_all-repmat(mean_images,1,size(data_train,2));
    %mean_subst_test=scale_all_test-repmat(mean_images,1,size(data_test,2));
    
%-------------------1 EXACT--------------%    
    %coeff = pca(mean_subst');
    %cov_scale=(mean_subst*mean_subst')*(1/size(data_train,2)); %(estimate of covariance)
    %[vector_temp, value]=eig(cov_scale);
    %vector_biggest=vector_temp(:,end-num_pca+1:end);
    %vector_ori=vector_biggest'*mean_subst;
    %vector_ori_test=vector_biggest'*mean_subst_test;

%-------------------2 TURK TRICK--------------%    

    %cov_scale=(mean_subst'*mean_subst)*(1/size(data_train,2)); %(estimate of covariance)
    %[vector_temp, value]=eig(cov_scale);
    %vector_biggest=vector_temp(:,end-num_pca+1:end);
    %temp=mean_subst*vector_biggest;
    %vector_ori = temp'*mean_subst;
    %vector_ori_test = temp'*mean_subst_test;

%------------------END--------------%    

 %data_train = vector_ori;
 %data_test = vector_ori_test;
 %s=std(data_train,[],2);
 %m=mean(data_train,2);
 %data_train=bsxfun(@minus, data_train, m);
 %data_train=bsxfun(@rdivide, data_train, s);
 %data_test=bsxfun(@minus, data_test, m);
 %data_test=bsxfun(@rdivide, data_test, s);
  
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
ei.layer_sizes = [30,ei.output_dim];
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
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost_regularized,...
    params,options,ei, data_train, labels_train);

%lr = 0.01;
%opt_params = sgd_dnn(params, ei, data_train, labels_train,lr,100,data_test, labels_test);
    
%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost_regularized( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f\n', acc_test);

count(i) = length(labels_test);
acc(i) = acc_test*count(i);

[~, ~, pred] = supervised_dnn_cost_regularized( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f\n', acc_train);


end

fprintf('All accuracy: %f\n', sum(acc)/sum(count));
