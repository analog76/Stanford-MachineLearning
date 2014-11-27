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
%







 

 
 J = sum(sum(R.* (X * Theta' - Y).^2))/2;







% =============================================================


%	X_grad = ((X*Theta'-Y)*Theta)(R == 1);

%	Theta_grad = ((X*Theta'-Y)'*X)(R == 1);

	X_grad = ((X * Theta' - Y) .* R) * Theta;		

   	
	
	Theta_grad=X' * ((X * Theta' - Y) .* R);
	
	X_grad=X_grad';
	Theta_grad=Theta_grad';


%	X
%	Theta
	%R
%	common = (X*Theta')-Y;
	
	%common
	
%	grad_temp = common .* R;
%	grad_temp
 
	
%	X_grad=grad_temp * Theta;
%	Theta_grad=X' * grad_temp;
%	X_grad
%	Theta_grad

	%X_grad_temp2 = X_grad_temp* Theta;
	
	%_grad_temp2
	
	
	%Theta_grad_temp = common .* R;
%	Theta_grad_temp
	
%	Theta_grad_temp2=Theta_grad_temp' * X;
	
%	Theta_grad_temp2
	
	 
	
	 
	 
	 %	 	 	 X_grad = ((X * Theta') - Y) * Theta;
		  
	%			 X_grad =  ((R'*X_grad));

   
		%heta_grad = ((X * Theta') - Y)'*X ;
		%Theta_grad =  (sum(R*Theta_grad));
		
	

grad = [X_grad(:); Theta_grad(:)];


		 
end
