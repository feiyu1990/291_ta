function [cost, theta] = sgd_dnn_validate(theta, ei, X, y, lr,check_size,X_validate, y_validate)
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
  
  stack = params2stack(theta, ei);
  numHidden = numel(ei.layer_sizes) - 1;
  hAct = cell(numHidden+1, 1);
  gradStack = cell(numHidden+1, 1);
  delta = cell(numHidden+1,1);
  lambda = ei.lambda;
  I = randperm(length(y));
  y=y(I); % labels in range 1 to 10
  X=X(:,I);
  
  prev_update = cell(numHidden+1, 1);
  update = cell(numHidden+1,1);
  momentum = ei.momentum;
  
  if ei.activation_fun == 'tanh'
    fun = @tanh_mine;
  else 
    fun = @sigmoid;
  end
  
%f_before = Inf;
iter = 0;
m = size(X,2);
%step = 1;
diff = inf;
prev = inf;
cost = 0;
count = 0;
max_count = 5;
max_iter = 100000;
stop = false;
best_param = theta;
best_acc = 0;
prev_acc = 0;
num_worse = 0;
max_worse = 100;
num_equal = 0;
best_tobe = 0;
second_besttobe = 0;
best_count = 0;
second_best = 0;

while(stop == false && iter < max_iter)
    i = rem(iter, m) + 1;
    data = X(:,i);
    labels = y(i);
    
    W = stack{1,1}.W;
    b = stack{1,1}.b;
    hAct{1} = fun(bsxfun(@plus, W*data, b));
    
    for i = 2:numHidden
        W = stack{i,1}.W;
        b = stack{i,1}.b;
        hAct{i} = fun(bsxfun(@plus, W*hAct{i-1}, b));
        %if hidden, then hAct is after nonlinear activation

    end
    W = stack{end,1}.W;
    b = stack{end,1}.b;
    hAct{end} = bsxfun(@plus,W*hAct{end-1},b);
    temp = exp(hAct{end});
    pred_prob = bsxfun(@rdivide, temp, sum(temp,1));

    %% compute cost
    %%% YOUR CODE HERE %%%
    I = sub2ind(size(pred_prob),labels',1:size(pred_prob,2));
    cost = -sum(log(pred_prob(I)));
    for i=1:numHidden + 1
        W = stack{i}.W;
        cost = cost + lambda*W(:)'*W(:);
    end
    
    %% compute gradients using backpropagation
    temp = pred_prob;
    temp(I) = temp(I) - 1;
    delta{end} = temp;
    for i = 1:numHidden
        W = stack{end - i + 1}.W;
        temp = W' * delta{end- i + 1};
        if ei.activation_fun == 'tanh'
            g_prime = 1 - hAct{end-i}.*hAct{end-i};
        else
            g_prime = (1 - hAct{end-i}).*hAct{end-i};
        end
        delta{end-i} = g_prime.*temp;
    end

    %% compute weight penalty cost and gradient for non-bias terms
    %%% YOUR CODE HERE %%%
    gradStack{1}.W = lr * (delta{1}*data' + 2*lambda*stack{1}.W);
    gradStack{1}.b = sum(delta{1},2);
    for i=2:numHidden+1
       gradStack{i}.W = delta{i}*hAct{i-1}' + 2*lambda*stack{i}.W ;
       gradStack{i}.b = sum(delta{i},2);
    end
    if iter>0
        for i=1:numHidden+1
           update{i}.W = momentum*prev_update{i}.W-lr*gradStack{i}.W;
           update{i}.b = momentum*prev_update{i}.b-lr*gradStack{i}.b;
        end    
    else
        for i=1:numHidden+1
           update{i}.W = -lr*gradStack{i}.W;
           update{i}.b = -lr*gradStack{i}.b;
        end
    end
    
    
    for i=1:numel(stack)
        stack{i}.W = stack{i}.W + update{i}.W;
        stack{i}.b = stack{i}.b + update{i}.b;
    end

    %if rem(iter, check_size/10)==0
    %    fprintf('Iteration: %d, Objective function: %f, lr: %e\n', iter, cost, lr);
    %end
    if rem(iter,check_size)==0
        [theta] = stack2params(stack);
        [~, ~, pred] = supervised_dnn_cost_regularized( theta, ei, X_validate, [], true);
        [~,pred] = max(pred);
        acc_test = mean(pred'==y_validate);     
        
        %I = sub2ind(size(pred),y_validate',1:size(pred,2));
        %acc_test = -sum(log(pred(I)));
        %for i=1:numHidden+1
        %    W = stack{i}.W;
        %    acc_test = acc_test + lambda*W(:)'*W(:);
        %end
        
        
        %if prev_acc > acc_test
        %    num_worse = num_worse + 1;
        %end
        if prev_acc == acc_test
            num_equal = num_equal + 1;
        else
            num_equal = 0;
        end
        
        
        if best_tobe < acc_test
            best_count = 0;
            second_besttobe = best_tobe;
            best_tobe = acc_test;
        end
        if best_tobe == acc_test
            best_count = best_count+1;
            
            if best_count == 10
                best_param = theta;
                best_acc = best_tobe;
                second_best = second_besttobe;
                num_worse = 0;
            end
        end
        if acc_test < second_best
            num_worse = num_worse+1;
        end
        
        
        prev_acc = acc_test;
        if num_worse >= max_worse || num_equal > 200
            stop = true;
        end
        
        [~, ~, pred_prob] = supervised_dnn_cost_regularized( theta, ei, X, [], true);
        I = sub2ind(size(pred_prob),y',1:size(pred_prob,2));
        cost = -sum(log(pred_prob(I)));
        for i=1:numHidden+1
            W = stack{i}.W;
            cost = cost + lambda*W(:)'*W(:);
        end
        fprintf('Iteration: %d, Objective function: %f, validate accuracy: %f, lr: %e\n', iter, cost, acc_test, lr);
        diff = prev - cost;
        if (diff <0)
            %if count == max_count
            %    count = 0;          
            %    lr = lr * 0.9;
            %else
            %    count = count + 1;
            %end
            diff = -diff;
        end
        if (rem(iter,40000)==0 && lr > 10e-5 && iter > 0)
            lr = lr * 0.5;
        end
        prev = cost;
        
    end
    
    iter = iter + 1;
    prev_update = update;
end
%[theta] = stack2params(stack);
theta = best_param;
end