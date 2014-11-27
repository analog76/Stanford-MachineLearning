function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%



 
 
val=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

val=[0.1; 0.01;  1];
 

%C_Arr=[0.01; 0.03];
%sigma_Arr=[0.01; 0.03]; 

t_error=zeros(length(val)*length(val), 3);

k=1;
for i=1:length(val)
	for j=1:length(val)
			k
			val(i)
			val(j)
			model= svmTrain(X, y, val(i), @(x1, x2) gaussianKernel(x1, x2, val(j))); 
 			predictions = svmPredict(model, Xval);  
 			t_error(k,1) = val(i); 
			t_error(k,2) = val(j); 
			t_error(k,3) = mean(double(predictions ~= yval)); 
			k=k+1;
	end

end





err = min(t_error(:,3));

 




err = min(t_error(:,3));


for i=1:length(t_error)
	s=t_error(i,3);
  if(err==s)
		C=t_error(i,1);
		sigma=t_error(i,2);
  end
end


 
 
% =========================================================================

end
