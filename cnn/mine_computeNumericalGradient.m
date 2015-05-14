function numgrad = mine_computeNumericalGradient(J, stack, number_check)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros

theta = stackToParams(stack);

numgrad = zeros(size(theta));
check_num = randperm(length(numgrad));
check_num = check_num(1:number_check);

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 
indices = [1];
for i =1:length(stack)
    curr = indices(end);
    size_curr = numel(stack{i}.W);
    indices = [indices, curr + size_curr];
end
for i =1:length(stack)
    curr = indices(end);
    size_curr = numel(stack{i}.b);
    indices = [indices, curr + size_curr];
end

epsilon = 1e-4;



count = 1;
for i =1:length(stack)
    check_list = randperm(numel(stack{i}.W));
    for j = 1:5
        curr_check = check_list(j);
        index = ind2sub(size(stack{i}.W), curr_check);
        oldT = stack{i}.W(index);
        stack{i}.W(index)=oldT+epsilon;
        pos = J(stack);
        stack{i}.W(index)=oldT-epsilon;
        neg = J(stack);
        numgrad(curr_check + indices(i) - 1) = (pos-neg)/(2*epsilon);
        stack{i}.W(index)=oldT;
        count = count + 1;
        if mod(count,10)==0
            fprintf('Done with %d\n',count);
        end
    end
end
for i =1:length(stack)
    check_list = randperm(numel(stack{i}.b));
    for j = 1:5
        index = check_list(j);
        oldT = stack{i}.b(index);
        stack{i}.b(index)=oldT+epsilon;
        pos = J(stack);
        stack{i}.b(index)=oldT-epsilon;
        neg = J(stack);
        numgrad(index + indices(i + length(stack)) - 1) = (pos-neg)/(2*epsilon);
        stack{i}.b(index)=oldT;
        count = count + 1;
        if mod(count,10)==0
            fprintf('Done with %d\n',count);
        end
    end
end




%% random number of checks
%for i =1:number_check
%    if mod(i,10)==0
%        fprintf('Done with %d\n',i);
%    end
%    curr_check = check_num(i);
%    for j =1:length(indices) - 1
%        if indices(j) > curr_check
%            break;
%        end
%    end
%    k = curr_check - indices(j - 1) + 1;
%        if (j - 1) <= length(stack)
%            stack_index = j-1;
%            index = ind2sub(size(stack{stack_index}.W), k);
%            oldT = stack{stack_index}.W(index);
%            stack{stack_index}.W(index)=oldT+epsilon;
%            pos = J(stack);
%            stack{stack_index}.W(index)=oldT-epsilon;
%            neg = J(stack);
%            numgrad(curr_check) = (pos-neg)/(2*epsilon);
%            stack{stack_index}.W(index)=oldT;
%        else
%            stack_index = j-5;
%            index = k;
%            oldT = stack{stack_index}.b(index);
%            stack{stack_index}.b(index)=oldT+epsilon;
%            pos = J(stack);
%            stack{stack_index}.b(index)=oldT-epsilon;
%            neg = J(stack);
%            numgrad(curr_check) = (pos-neg)/(2*epsilon);
%            stack{stack_index}.b(index)=oldT;
%        end
%    
%end


%% original    
%epsilon = 1e-4;
%count = 1;
%for i =1:length(stack)
%    for j=1:numel(stack{i}.W);
%        index = ind2sub(size(stack{i}.W), j);
%        oldT = stack{i}.W(index);
%        stack{i}.W(index)=oldT+epsilon;
%        pos = J(stack);
%        stack{i}.W(index)=oldT-epsilon;
%        neg = J(stack);
%        numgrad(count) = (pos-neg)/(2*epsilon);
%        stack{i}.W(index)=oldT;
%        count = count + 1;
%        if mod(count,100)==0
%            fprintf('Done with %d\n',count);
%        end
%    end
%end
%for i =1:length(stack)
%    for j=1:length(stack{i}.b);
%        oldT = stack{i}.b(j);
%        stack{i}.b(j)=oldT+epsilon;
%        pos = J(stack);
%        stack{i}.b(j)=oldT-epsilon;
%        neg = J(stack);
%        numgrad(count) = (pos-neg)/(2*epsilon);
%        stack{i}.b(j)=oldT;
%        count = count + 1;
%        if mod(count,100)==0
%            fprintf('Done with %d\n',count);
%        end
%    end
%end
    

%% original UFLDL

%for i =1:length(numgrad)
%    oldT = theta(i);
%    theta(i)=oldT+epsilon;
%    pos = J(theta);
%    theta(i)=oldT-epsilon;
%    neg = J(theta);
%    numgrad(i) = (pos-neg)/(2*epsilon);
%    theta(i)=oldT;
%    if mod(i,100)==0
%       fprintf('Done with %d\n',i);
%    end;
%end;





%% ---------------------------------------------------------------
end
