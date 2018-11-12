function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Need to compute two layers, a1 and a2

% X has dim mx401 (m is num of examples)
column1s = ones(m,1);
X = [column1s X];

% z(2) has dimensions 25xm
z2 = Theta1*transpose(X);
a2 = sigmoid(z2);
% Add column of 1's to a2 so theta2*a2 is possible
a2 = transpose(a2);
a2 = [column1s a2];
a2 = transpose(a2);
% z3 has dimensions 10xm
z3 = Theta2*a2;
% z3 now has dimensions mx10
z3 = transpose(z3);

[dummy,p] = max(z3,[],2);



% =========================================================================


end
