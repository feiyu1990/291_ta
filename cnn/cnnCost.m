function [cost, grad, preds] = cnnCost(ei, theta,images,labels,numClasses,...
                                pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width of image
numImages = size(images,4); % number of images
numChannels = size(images,3); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias

layer_type = ei.layer_type;
layer_param = ei.layer_param;
numHidden = length(layer_type);
hAct = cell(numHidden, 1);
lambda = ei.lambda;
poolDim = layer_param{3}.pool_size;
localDim = layer_param{3}.local_size;
filterDim = layer_param{1}.filterDim;
poolType = layer_param{3}.type;

numFilters = layer_param{1}.numFilters;

[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numChannels, numFilters,...
                        poolDim,numClasses);
                    



count = 0;
for i = numHidden:-1:1
    %fully connected
    if strcmp(layer_type{i} , 'full') || strcmp(layer_type{i} , 'conv')
        count = count + 1;
    end
end
stack = cell(count,1);
gradStack = cell(count,1);

stack{1}.W = Wc;
stack{2}.W = Wd;
stack{1}.b = bc;
stack{2}.b = bd;



% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations
activations = zeros(convDim,convDim,numFilters,numImages);
activations = cnnConvolve_mine(filterDim, numFilters, images, Wc, bc);
hAct{1} = activations;

if strcmp(ei.activation_fun , 'tanh')
    activations = tanh_mine(activations);
elseif strcmp(ei.activation_fun , 'relu')
    activations = max(activations,0);
else
    activations = sigmoid(activations);
end
hAct{2} = activations;
    

        

% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations
%activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);
[activationsPooled,mask] = cnnPool(poolDim, localDim, activations, poolType);
hAct{3} = activationsPooled;

%%% YOUR CODE HERE %%%

% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
activationsPooled = reshape(activationsPooled,[],numImages);
%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.
probs = zeros(numClasses,numImages);

temp = exp(bsxfun(@plus, Wd*activationsPooled, bd));
hAct{4} = probs;

probs = bsxfun(@rdivide, temp, sum(temp,1));
I = sub2ind(size(probs),labels',1:size(probs,2));


%%% YOUR CODE HERE %%%

%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

%cost = 0; % save objective into cost
cost = -sum(log(probs(I)));
cost = cost/numImages;
for i=1:length(stack)
    W = stack{i}.W;
    cost = cost + lambda*W(:)'*W(:);
end



%%% YOUR CODE HERE %%%

% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.

%%% YOUR CODE HERE %%%



delta = cell(numHidden+1,1);

temp = probs;
temp(I) = temp(I) - 1;
delta{end} = temp;
w_count = 0;




%last fully connected
for i = numHidden:-1:2
    %fully connected
    if strcmp(layer_type{i} , 'full')
        W = stack{end-w_count}.W;
        b = stack{end-w_count}.b;
        w_count = w_count + 1;
        temp = W' * delta{i+1};

        delta{i} = temp;
    end
    
    %pooling
    if strcmp(layer_type{i} , 'pool')
        [a,b,c,d] = size(hAct{i});
        if length(size(delta{i+1})) == 2
            %should reshape it
            temp = reshape(delta{i+1},a,b,c,d);
        else
            temp = delta{i+1};
        end
        this_poolDim = layer_param{i}.pool_size;
        %not used right now
        this_localDim = layer_param{i}.local_size;
        
        delta{i} = zeros(size(hAct{i-1}));
        
        if strcmp(layer_param{i}.type, 'mean')
            for j=1:c
                for k=1:d
                     delta{i}(:,:,j,k) = (1/this_poolDim^2) * kron(temp(:,:,j,k),ones(this_poolDim));
                end
            end
        else
            for j=1:c
                for k=1:d
                     delta{i}(:,:,j,k) = mask(:,:,j,k).*kron(temp(:,:,j,k),ones(this_poolDim));
                end
            end
        end
    end
    
    if strcmp(layer_type{i} , 'nonlinear')
        if strcmp(ei.activation_fun , 'tanh')
            g_prime = 1 - hAct{i}.*hAct{i};
        elseif strcmp(ei.activation_fun , 'relu')
            g_prime = double(hAct{i} > 0);
        else 
            g_prime = (1 - hAct{i}).*hAct{i};
        end
        delta{i} = delta{i+1}.*g_prime;
    end
    
    %convutional layer
    if strcmp(layer_type{i}, 'conv')
        W = stack{end-w_count}.W;
        b = stack{end-w_count}.b;
        w_count = w_count + 1;
        
        if i==1
            this_input = images;
        else
            this_input = hAct{i};
        end
        [width, height, channels, num] = size(this_input);
        delta{i} = zeros(size(this_input));
        for imageNum = 1:num
            for ch = 1:channels
                convolvedImage = zeros(width, height);
                for outChannel = 1:size(delta{i+1},3)
                    filter = W(:,:,ch,outChannel);
                    im = squeeze(delta{i+1}(:, :,ch, outChannel));
                    convolvedImage = convolvedImage + conv2(im, filter, 'full');
                end
                delta{i}(:, :, ch, imageNum) = convolvedImage;
            end
        end
    end
end




%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

count = length(stack);
for i = numHidden:-1:1
    %fully connected
    if i == 1
        this_input = images;
    else
        this_input = hAct{i-1};
    end
    if strcmp(layer_type{i} , 'full')
        if length(size(this_input)) >2
            temp_act = reshape(this_input,[],numImages);
        else
            temp_act = this_input;
        end
        gradStack{count}.W = (delta{i+1}*temp_act')/numImages + 2*lambda*stack{count}.W ;
        gradStack{count}.b = (sum(delta{i+1},2))/numImages;
        count = count - 1;
        
    elseif strcmp(layer_type{i} , 'conv')
        gradStack{count}.W = zeros(size(stack{count}.W));
        [width,height,numInput,numFilter] = size(stack{count}.W);
        for inputNum = 1:numInput 
            for filterNum = 1:numFilter
                convolvedImage = zeros(width, height);
                for imageNum = 1:numImages
                    filter = delta{i+1}(:,:,filterNum,imageNum);
                    filter = rot90(squeeze(filter),2);
                    im = squeeze(this_input(:, :,inputNum, imageNum));
                    convolvedImage = convolvedImage + conv2(im, filter, 'valid');
                end
                gradStack{count}.W(:,:,inputNum, filterNum) = convolvedImage/numImages;
            end
        end
        gradStack{count}.W = gradStack{count}.W + 2*lambda*stack{i}.W;
        gradStack{count}.b = (squeeze(sum(sum(sum(delta{i+1},1),2),4)))/numImages;        
        count = count - 1;
    end
end


%% Unroll gradient into grad vector for minFunc
Wc_grad = gradStack{1}.W;
Wd_grad = gradStack{2}.W;
bc_grad = gradStack{1}.b;
bd_grad = gradStack{2}.b;

grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

end
