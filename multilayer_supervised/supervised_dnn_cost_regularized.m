function [ cost, grad, pred_prob] = supervised_dnn_cost_regularized( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
delta = cell(numHidden+1,1);
lambda = ei.lambda;
if ei.activation_fun == 'tanh'
    fun = @tanh_mine;
elseif ei.activation_fun == 'relu'
    fun = @(x) max(0,x);
else
    fun = @sigmoid;
end
%% forward prop
%%% YOUR CODE HERE %%%
if(numHidden>0)
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
else
    W = stack{end,1}.W;
    b = stack{end,1}.b;
    hAct{end} = bsxfun(@plus,W*data,b);
    temp = exp(hAct{end});
    pred_prob = bsxfun(@rdivide, temp, sum(temp,1));

end

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
I = sub2ind(size(pred_prob),labels',1:size(pred_prob,2));
cost = -sum(log(pred_prob(I)));
for i=1:numHidden+1
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
    elseif ei.activation_fun == 'relu'
            g_prime = double(hAct{end-i} > 0);
    else 
        g_prime = (1 - hAct{end-i}).*hAct{end-i};
    end
    delta{end-i} = g_prime.*temp;
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
gradStack{1}.W = delta{1}*data' + 2*lambda*stack{1}.W;
gradStack{1}.b = sum(delta{1},2);
for i=2:numHidden+1
    gradStack{i}.W = delta{i}*hAct{i-1}' + 2*lambda*stack{i}.W ;
    gradStack{i}.b = sum(delta{i},2);
end

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



