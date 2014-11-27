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

 

 
 
prediction  = 	 sigmoid(X * theta);

 
for i=1:rows(X)
	penalty0=-y(i).* log(prediction(i,:));
	penalty1=(1-y(i)) .* log(1-(prediction(i,:)));
	J=J+(penalty0-penalty1); 
end;
J=J/(m);


RegWeight=0;

for j=2:size(theta)
	RegWeight = RegWeight + theta(j)*theta(j);
end;
	RegWeight = RegWeight * lambda/(2*m); 

%J = J + RegWeight; 
 

J= (-y' * log(prediction)-(1-y)' * log(prediction) + (lambda/2) * (theta' * theta))/m;


%%% cODE FOR GRADIENT DESCENT

grad=(X'*(prediction-y))/m;
 
RegW = (lambda * theta)/m;
RegW(1) = 0;
%grad=grad + RegW;


 grad=(X' * (prediction-y)  + lambda * theta) /m;


 
 
% =============================================================

end
