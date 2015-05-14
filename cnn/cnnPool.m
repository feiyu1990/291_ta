function [pooledFeatures,mask] = cnnPool(poolDim, localDim, convolvedFeatures, option)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     
if ~exist('option','var')
    option = 'max';
end;
numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);
mask = zeros(convolvedDim, ...
        convolvedDim, numFilters, numImages);
    
    
% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.

%%% YOUR CODE HERE %%%



%% version 0.2 - a little quicker

%temp_patch = zeros(poolDim, poolDim, numFilters, numImages);



%temp_convolvedFeatures = reshape(convolvedFeatures, convolvedDim, convolvedDim, []);
%temp_pooledFeatures = zeros(convolvedDim / poolDim, ...
%        convolvedDim / poolDim, numFilters*numImages);
%temp_1mask = zeros(convolvedDim, ...
%        convolvedDim, numFilters * numImages);
    
%for i =1:numFilters*numImages
%    %for j=1:numImages
%        temp_patch = temp_convolvedFeatures(:,:,i);
%        patch_col = im2col(temp_patch,[poolDim, poolDim],'distinct');
%        patch_col=patch_col+rand(size(patch_col))*1e-12;
%        
%        if strcmp(option, 'mean')
%            temp = mean(patch_col,1);
%            temp_pooledFeatures(:,:,i) = col2im(temp, [1,1],[convolvedDim/poolDim, convolvedDim/poolDim],'distinct');
%        else
%            temp = max(patch_col,[],1);
%            temp_mask = patch_col ==  repmat(temp,size(patch_col,1),1);
%            temp_pooledFeatures(:,:,i) = col2im(temp, [1,1],[convolvedDim/poolDim, convolvedDim/poolDim],'distinct');
%            temp_1mask(:,:,i) = col2im(temp_mask, [poolDim,poolDim],[convolvedDim, convolvedDim],'distinct');
%        end
%    %end
%end
%pooledFeatures = reshape(temp_pooledFeatures, convolvedDim / poolDim, ...
%        convolvedDim / poolDim, numFilters, numImages);
%mask = reshape(temp_1mask, convolvedDim, ...
%        convolvedDim, numFilters, numImages);
%end


%% version 0.1 - all slow
for i =1:numFilters
    for j=1:numImages
        temp_patch = convolvedFeatures(:,:,i,j);
        patch_col = im2col(temp_patch,[poolDim, poolDim],'distinct');
        patch_col=patch_col+rand(size(patch_col))*1e-12;
        
        if strcmp(option, 'mean')
            temp = mean(patch_col,1);
            pooledFeatures(:,:,i,j) = col2im(temp, [1,1],[convolvedDim/poolDim, convolvedDim/poolDim],'distinct');
        else
            temp = max(patch_col,[],1);
            temp_mask = patch_col ==  repmat(temp,size(patch_col,1),1);
            pooledFeatures(:,:,i,j) = col2im(temp, [1,1],[convolvedDim/poolDim, convolvedDim/poolDim],'distinct');
            mask(:,:,i,j) = col2im(temp_mask, [poolDim,poolDim],[convolvedDim, convolvedDim],'distinct');
        end
    end
end
end

%% old version - max pooling very slow
%for i=1:size(pooledFeatures, 1)
%    for j=1:size(pooledFeatures, 2)
%        temp_patch = convolvedFeatures((i-1)*poolDim+1:min((i-1)*poolDim+localDim, convolvedDim),...
%                                       (j-1)*poolDim+1:min((j-1)*poolDim+localDim, convolvedDim), :, :);
%        
%        if strcmp(option, 'mean')
%            pooledFeatures(i,j,:,:) = mean(mean(temp_patch, 1), 2);
%        else
%            
%            pooledFeatures(i,j,:,:) = max(max(temp_patch,[], 1),[], 2);
%            for ii=1:numFilters
%                for jj=1:numImages
%                    M = squeeze(temp_patch(:,:,ii,jj));
%                    [max_val,idx]=max(M(:));
%                    [row,col]=ind2sub(size(M),idx);
%                    mask((i-1)*poolDim+row,(j-1)*poolDim+col,ii,jj) = mask((i-1)*poolDim+row,(j-1)*poolDim+col,ii,jj)+1;
%                end
%            end
%        end
%    end
%end
%end

