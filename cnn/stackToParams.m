function [ param ] = stackToParams( stack )
%STACKTOPARAMS Summary of this function goes here
%   Detailed explanation goes here

param = [];
for i=1:length(stack)
    param = [param; stack{i}.W(:)];
end
for i=1:length(stack)
    param = [param; stack{i}.b(:)];
end
end

