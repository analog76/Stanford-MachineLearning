function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% 
% Note: grad should have the same dimensions as theta
%


prediction  = 	 sigmoid(X * theta);




 
penaltya=-y * log(prediction);


penaltyb=-(1-y) * log(prediction);

J=(penaltya-penaltyb)/m;

for i=1:rows(X)
	penalty0=-y(i).* log(prediction(i,:));
	penalty1=(1-y(i)) .* log(1-(prediction(i,:)));
	J=J+(penalty0-penalty1); 
end;

J=J/m;


 
 

 
 grad=X' * (prediction-y) /m;
for j=1:size(theta)
%	temp = sum((prediction-y).* X(:,j));
%	grad(j) =  temp/m; 
end 
 
  


% =============================================================

end
