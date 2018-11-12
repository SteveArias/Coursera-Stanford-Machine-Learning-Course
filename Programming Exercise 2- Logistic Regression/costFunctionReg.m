function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%{ 
z = X*theta;
temp = (-transpose(y)*log(sigmoid(z))) - (transpose(1-y)*log(1-sigmoid(z)));
J = sum(temp)/m;
for j = 1:size(theta)
    temp2 = (sigmoid(z) - y).*X(:,j);
    grad(j,1) = sum(temp2)/m;
%}
z = X*theta;
temp = (-transpose(y)*log(sigmoid(z))) - (transpose(1-y)*log(1-sigmoid(z)));
thetasqr = zeros(size(theta)-1);
for x = 1:size(theta)-1
    thetasqr(x,1) = theta(x+1,1).^2;
end
    
temp2 = (lambda/(2*m))*sum(thetasqr);
J = (sum(temp)/m) + temp2;

for j = 1:size(theta)
    temp2 = (sigmoid(z) - y).*X(:,j);
    grad(j,1) = sum(temp2)/m;
    if j ~= 1
        grad(j,1) = grad(j,1) +(lambda/m)*theta(j,1);
    end
end







% =============================================================

end
