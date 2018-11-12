function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));
regThetaTemp = 0;
regXTemp = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta

for i=1:num_movies
    idx = find(R(i,:)==1);
    %disp("idx: " + idx + " for movie #" + i);
    theta_temp = Theta(idx, :);
    %disp("theta_temp: "+ theta_temp + " for movie # " + i);
    Y_temp = Y(i,idx);
    %disp("Y_temp: "+ Y_temp + " for movie # " + i);
    X_grad(i,:) = (X(i,:)*transpose(theta_temp) - Y_temp)*theta_temp + (lambda*X(i,:));
       
    for j=1:num_users
        % This is a vector matrix on what movies this user has rated
        idy=find(R(:,j)==1);
        % This takes the features of the movies this user has rated
        X_temp = X(idy,:);
        % This takes the ratings of the movies this user has rated
        Y_temp = Y(idy,j);
        % This takes the theta params of the current user
        theta_temp = Theta(idx, :);

        %disp("X_temp");
        %disp(size(X_temp));
        
        %disp("transpose(Theta(j,:))");
        %disp(size(transpose(Theta(j,:))));
        
        %disp("Y_temp");
        %disp(size(Y_temp));
        
        %disp("X_temp");
        %disp(size(X_temp));
        
        Theta_grad(j,:)= transpose(transpose(X_temp*transpose(Theta(j,:)) - Y_temp)*X_temp) + transpose(lambda*Theta(j,:));
        
        if R(i,j)==1
            J = J + ((X(i,:)*transpose(Theta(j,:)))-Y(i,j))^2;
            
        end
    end
end
J=J/2;

for i=1:num_users
    for j=1:num_features
        regThetaTemp = regThetaTemp + (Theta(i,j)^2);
    end
end

regThetaTemp = regThetaTemp * (lambda / 2);

for i=1:num_movies
    for j=1:num_features
        regXTemp = regXTemp + (X(i,j)^2);
    end
end

regXTemp = regXTemp * (lambda / 2);

J = J + regThetaTemp + regXTemp;






% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
