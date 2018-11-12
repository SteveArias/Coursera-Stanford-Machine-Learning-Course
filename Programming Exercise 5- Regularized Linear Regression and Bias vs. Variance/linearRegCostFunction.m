function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% z is dimension 12 x 1
z = X*theta;
temp = (z-y).^2;

temp1 = sum(temp);
temp2 = temp1/(2*m);

regTheta = theta;
regTheta(1,1) = 0;

blah = 2*m;
blahblah = lambda/blah;

regTemp = sum(regTheta.^2)*blahblah;
J = temp2 + regTemp;

grad = transpose(X)*(z-y);
grad = grad/m;

gradTheta = theta;
gradTheta(1,1) = 0;
grad = grad + (gradTheta.*lambda)/m;
















% =========================================================================

grad = grad(:);

end
