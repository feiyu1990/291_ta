function [ image_matrix ] = load_vwfa( image_cell, resize_size )
%LOAD_VWAF Summary of this function goes here
%   Detailed explanation goes here

temp = imread(image_cell{1});
%width = size(temp,1);
image_matrix = zeros(resize_size, resize_size, length(image_cell));

for i=1:length(image_cell)
    temp = imread(image_cell{i});
    temp = rgb2gray(im2double(temp));
    temp = imresize(temp, [resize_size, resize_size]);
    image_matrix(:,:,i) = temp;
end
end

