%
%This exercise uses a data from the UCI repository:
% Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository
% http://archive.ics.uci.edu/ml
% Irvine, CA: University of California, School of Information and Computer Science.
%
%Data created by:
% Harrison, D. and Rubinfeld, D.L.
% ''Hedonic prices and the demand for clean air''
% J. Environ. Economics & Management, vol.5, 81-102, 1978.
%
close all;clear all;clc;
addpath ../common
addpath ../common/minFunc_2012/minFunc
addpath ../common/minFunc_2012/minFunc/compiled
% Load housing data from file.
data = load('housing.data');
data=data'; % put examples in columns

% Include a row of 1s as an additional intercept feature.

% Shuffle examples.
data = data(:, randperm(size(data,2)));

% Split into train and test sets
% The last row of 'data' is the median home price.

train.X = data(1:end-1,1:400);
train.y = data(end,1:400);


test.X = data(1:end-1,401:end);
test.y = data(end,401:end);


%%%normalize so that [-1,1]%%%
%min_x = min(train.X,[],2);
%max_x = max(train.X,[],2);
%train.X = bsxfun(@rdivide, bsxfun(@minus, train.X, min_x), max_x-min_x)*2-1;
%test.X = bsxfun(@rdivide, bsxfun(@minus, test.X, min_x), max_x-min_x)*2-1;


%%%normalize so that 0 mean unit variance%%%
s=std(train.X,[],2);
m=mean(train.X,2);
train.X=bsxfun(@minus, train.X, m);
train.X=bsxfun(@rdivide, train.X, s+.1);
test.X=bsxfun(@minus, test.X, m);
test.X=bsxfun(@rdivide, test.X, s+.1);
  

train.X = [ ones(1,size(train.X,2)); train.X ];
test.X = [ ones(1,size(test.X,2)); test.X ];

m=size(train.X,2);
n=size(train.X,1);

% Initialize the coefficient vector theta to random values.
theta = rand(n,1);

% Run the minFunc optimizer with linear_regression.m as the objective.
%
% TODO:  Implement the linear regression objective and gradient computations
% in linear_regression.m
%

lr = 0.00001;
alpha = 2;
theta_gd = linear_regression_mine(theta, train.X, train.y, lr);
%theta_gd = linear_regression_backtrack(theta, train.X, train.y, alpha);

%theta_cf = linear_regression_newton(theta, train.X, train.y);


options = struct('MaxIter', 200);
theta_mf = minFunc(@linear_regression_vec, theta, options, train.X, train.y);
%fprintf('Optimization took %f seconds.\n', toc);

theta_cf = lr_gradient(train.X,train.y);

% Run minFunc with linear_regression_vec.m as the objective.
%
% TODO:  Implement linear regression in linear_regression_vec.m
% using MATLAB's vectorization features to speed up your code.
% Compare the running time for your linear_regression.m and
% linear_regression_vec.m implementations.
%
% Uncomment the lines below to run your vectorized code.
%Re-initialize parameters
%theta = rand(n,1);
%tic;
%theta = minFunc(@linear_regression_vec, theta, options, train.X, train.y);
%fprintf('Optimization took %f seconds.\n', toc);

% Plot predicted prices and actual prices from training set.
actual_prices = train.y;
predicted_prices_mf = theta_mf'*train.X;
predicted_prices_gd = theta_gd'*train.X;
predicted_prices_cf = theta_cf'*train.X;


% Print out root-mean-squared (RMS) training error.
train_rms=sqrt(mean((predicted_prices_mf - actual_prices).^2));
fprintf('MinFunc: RMS training error: %f\n', train_rms);
train_rms=sqrt(mean((predicted_prices_gd - actual_prices).^2));
fprintf('Gradient Descent: RMS training error: %f\n', train_rms);
train_rms=sqrt(mean((predicted_prices_cf - actual_prices).^2));
fprintf('Closed form: RMS training error: %f\n', train_rms);


% Print out test RMS error
actual_prices = test.y;
predicted_prices_mf = theta_mf'*test.X;
predicted_prices_gd = theta_gd'*test.X;
predicted_prices_cf = theta_cf'*test.X;


test_rms=sqrt(mean((predicted_prices_mf - actual_prices).^2));
fprintf('MinFunc: RMS testing error: %f\n', test_rms);
test_rms=sqrt(mean((predicted_prices_gd - actual_prices).^2));
fprintf('Gradient Descent: RMS testing error: %f\n', test_rms);
test_rms=sqrt(mean((predicted_prices_cf - actual_prices).^2));
fprintf('Closed form: RMS testing error: %f\n', test_rms);



% Plot predictions on test data.
plot_prices=true;
if (plot_prices)
  figure;
  [actual_prices,I] = sort(actual_prices);
  predicted_prices_mf=predicted_prices_mf(I);
  predicted_prices_gd=predicted_prices_gd(I);
  predicted_prices_cf=predicted_prices_cf(I);

  plot(actual_prices, 'rx');
  hold on;
  plot(predicted_prices_mf,'bx');
  hold on;
  plot(predicted_prices_cf,'g');
  plot(predicted_prices_gd,'cx');
  %legend('Actual Price', 'Predicted Price', 'Closed form');
  xlabel('House #');
  ylabel('House price ($1000s)');
  
 

end