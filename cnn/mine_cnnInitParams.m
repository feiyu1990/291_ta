function stack = mine_cnnInitParams(imageDim, numInput, ei)
% Initialize parameters for a single layer convolutional neural
% network followed by a softmax layer.
%                            
% Parameters:
%  imageDim   -  height/width of image
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  numClasses -  number of classes to predict
%
%
% Returns:
%  theta      -  unrolled parameter vector with initialized weights

%% Initialize parameters randomly based on layer sizes.

input_dim = [imageDim, imageDim, numInput];
count = 1;

for i=1:length(ei.layer_param)
    param = ei.layer_param{i};
    if strcmp(ei.layer_type{i}, 'conv')
        filterDim = param.filterDim;
        numFilters =param.numFilters;
        numChannels = param.numInput;
        assert(numChannels == input_dim(3), 'CONV: input dimension doesnt match!');
        outDim = input_dim(1) - filterDim + 1;
        W = 1e-1*randn(filterDim,filterDim,numChannels, numFilters);
        b = zeros(numFilters, 1);
        stack{count}.W = W;
        stack{count}.b = b;
        count = count + 1;
        input_dim = [outDim, outDim, numFilters];
    elseif strcmp(ei.layer_type{i}, 'pool')
        poolDim = param.pool_size;
        assert(mod(input_dim(1), poolDim)==0, ...
            'poolDim must divide imageDim - filterDim + 1');
        outDim = input_dim(1) / poolDim;
        input_dim = [outDim, outDim, input_dim(3)];
    elseif strcmp(ei.layer_type{i}, 'full')
        if length(input_dim) > 1
            input_dim = [input_dim(1)*input_dim(2)*input_dim(3)];
        end
        outDim = param.outputDim;
        r  = sqrt(6) / sqrt(outDim+input_dim(1)+1);
        W = rand(outDim, input_dim(1)) * 2 * r - r;
        b = zeros(outDim, 1);
        stack{count}.W = W;
        stack{count}.b = b;
        count = count + 1;
        input_dim = [outDim];
    end
        

end

