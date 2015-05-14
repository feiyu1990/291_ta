function [theta] = logistic_regression_mine(theta,X,y,lr)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example.  y(j) is the target for example j.
  %
  
  m=size(X,2);
  n=size(X,1);

  
  
  f_before = Inf;
  iter = 0;
  step = 1;
  while(abs(step) >= 10e-7)
      hx = 1./(1+exp(-theta'*X));
      f = -y*log(hx)'-(1-y)*log(1-hx)';
      g = X*(hx-y)';
      g = sum(g, 2)/m;
      update = g*lr;
      step = f_before - f;
      if (step < 0)
          lr = lr / 2;
      end
      f_before = f;
      if rem(iter, 10)==0
        fprintf('Iteration: %d, Objective function: %f, lr: %e\n', iter, f, lr);
      end
      theta = theta - update;
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
