function [theta] = softmax_regression_gd(theta, X, y, lr)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);
  
  I = randperm(length(y));
  y=y(I); % labels in range 1 to 10
  X=X(:,I);
  
  num_classes=size(theta,2)+1;

  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
  theta = cat(2,theta, zeros(n,1));
 

%f_before = Inf;
iter = 0;
%step = 1;
while(iter < 100*m)
    i = rem(iter, m) + 1;
    x_point = X(:,i);
    y_point = y(i);
    hx = bsxfun(@rdivide, exp(theta'*x_point), sum(exp(theta'*x_point), 1));
    I= sub2ind(size(hx), y_point, 1:size(hx,2));
    f = - log(hx(I));
    temp = -hx;
    temp(I) = temp(I) + 1;
    g = -x_point*temp';
    g = g(:,1:num_classes-1);
    update = g*lr;
    %step = f_before - f;
    %if (step < 0)
    %    lr = lr / 2;
    %end
    %f_before = f;
    if rem(iter, 100)==0
        fprintf('Iteration: %d, Objective function: %f, lr: %e\n', iter, f, lr);
    end
    theta = theta - cat(2, update, zeros(n,1));
    iter = iter + 1;
end
