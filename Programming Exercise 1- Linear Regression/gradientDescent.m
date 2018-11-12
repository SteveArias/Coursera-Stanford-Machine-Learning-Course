function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    %temp1 = (X(:,1)*theta(1,1)-y).*X(:,1); %mby1 1 by 1
    %temp2 = (X(:,2)*theta(2,1)-y).*X(:,2);
    temp = (X*theta - y);
    temp1 = temp.*X(:,1);
    temp2 = temp.*X(:,2);
    temp1 = sum(temp1);
    temp2 = sum(temp2);

    theta(1,1) = theta(1,1) - (alpha/m)*temp1;
    theta(2,1) = theta(2,1) - (alpha/m)*temp2;





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
