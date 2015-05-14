function [theta] = linear_regression_backtrack(theta,X,y, alpha)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example.  y(j) is the target for example j.
  %
  
  m=size(X,2);
  n=size(X,1);
  
  gamma = 0.5;
  c = 0.5;
  eps = 10e-7;
  
  
  f_before = Inf;
  iter = 0;
  step = 1;
  f = theta'*X - y;
  f = 1/2*(f*f');
  %while(abs(step) >= 10e-5)
  while(abs(step) >= 10e-5)
      g = X*(theta'*X - y)';
      g = g/m;
      [theta, f, fncall] =  backtrack(theta, X, y, -g/norm(g)*alpha, f, -norm(g)*alpha, c, gamma, eps);
      step = f_before - f;
      f_before = f;
      if rem(iter, 100)==0
        fprintf('Iteration: %d, Objective function: %f, number_call: %d\n', iter, f, fncall);
      end
      iter = iter + 1;
  end

  %
  % TODO:  Compute the linear regression objective by looping over the examples in X.
  %        Store the objective function value in 'f'.
  %
  % TODO:  Compute the gradient of the objective with respect to theta by looping over
  %        the examples in X and adding up the gradient for each example.  Store the
  %        computed gradient in 'g'.
  
%%% YOUR CODE HERE %%%
