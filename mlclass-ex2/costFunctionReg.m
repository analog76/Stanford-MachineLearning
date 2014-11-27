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

 

 
% Prediction Code 
% Implementation section are here;
% http://www.ml-class.org/course/qna/view?id=1648&page=3


% Regularization code;
prediction  = 	 sigmoid(X * theta);

 
Ja = -y' * log(prediction) -((1-y)' * (log(1-(prediction))));
Ja= Ja/m;

theta2=theta(2:size(theta),:);
theta0=[0];
thetaR = [theta0;theta2];
Jb = (thetaR' * thetaR)  * lambda/ (2*m);


J = Ja+Jb;


% Gradient Descent code

grad=(X' * (prediction-y)  + lambda * thetaR) /m;


 
 
% =============================================================

end
