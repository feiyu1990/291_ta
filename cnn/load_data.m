function [ image_matrix, label_batch ] = load_data( images, labels, batch_size, start_id, resize_size )
%LOAD_VWAF Summary of this function goes here
%   Detailed explanation goes here

temp = imread(images{1});
%width = size(temp,1);
image_matrix = zeros(resize_size, resize_size, 1, batch_size);
label_batch = labels(start_id: start_id + batch_size - 1);
for i=1:batch_size
    temp = imread(images{i + start_id - 1});
    temp = rgb2gray(im2double(temp));
    temp = imresize(temp, [resize_size, resize_size]);
    image_matrix(:,:,:,i) = temp;
end
end


