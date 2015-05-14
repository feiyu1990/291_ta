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

%for i=2:2
for i=1:length(identity_unique)
    %name = strcat(identity_unique(i), '_new_POFA_PreprocessedData_expression_30.mat');
    name = strcat(identity_unique(i), '_POFA_GaborData_expression.mat');
    load (name{1})
    

data_train = [ones(1,size(data.train,2)); data.train]; 
data_test = [ones(1,size(data.test,2)); data.test];

    labels_train = data.train_label;
    labels_test = data.test_label;


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



theta = rand(size(data_train,1),max(labels_train))*0.1;


options = struct('MaxIter', 2000);
theta(:)=minFunc(@softmax_regression_vec, theta(:), options, data_train, labels_train');


%lr = 0.01;
%opt_params = sgd_dnn(params, ei, data_train, labels_train,lr,100,data_test, labels_test);
    
%% compute accuracy on the test and train set
acc_test = multi_classifier_accuracy(theta,data_test,labels_test');

fprintf('test accuracy: %f\n', acc_test);

count(i) = length(labels_test);
acc(i) = acc_test*count(i);
acc_train = multi_classifier_accuracy(theta,data_train,labels_train');
fprintf('train accuracy: %f\n', acc_train);


end

fprintf('All accuracy: %f\n', sum(acc)/sum(count));
