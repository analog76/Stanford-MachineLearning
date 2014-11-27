function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
 
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



a1=[ones(m,1) X];

a2=sigmoid(a1 * Theta1');

a2=[ones(m,1)  a2];

a3=sigmoid(a2 * Theta2');

a3;

z = eye(num_labels); 

Y=zeros(m,num_labels);

for i = 1 :m
	Y(i,:)=z(y(i),:);

end


size(a3);
 
 for i = 1:m 
	for j = 1:num_labels 
		%v = z(i,:); 
		% Select the ith row of 1's and 0's vector 
		%temp = (-Y(i,j) * log(a3(i,j)) -(1-Y(i,j))*log(1-a3(i,j)))/m;
		%log(a3(i,j))
		%temp = -Y(i,j) * log(a3(i,j)');
		%Y(i,j)*log(a3(i,j))		
		J = J +  (-Y(i,j) * log(a3(i,j)) -(1-Y(i,j))*log(1-a3(i,j)))/m;
		
		%J = J + (1/m) sum(-v . log(a3(:,j)')-(1-vect) .* log(1-a3(:,j)'));
	end 
end

 
 
 

Theta1Size=size(Theta1);%Theta1(:,2:length(Theta1)));
a2j=Theta1Size(1:1);
a2k=Theta1Size(2:2);

Theta2Size=size(Theta2);%(:,2:length(Theta2)));
a3j=Theta2Size(1:1);
a3k=Theta2Size(2:2);


a2j;
a2k;
a3j;
a3k;

% size(Theta1(:,2:length(Theta1))
Theta1Square =0;
Theta2Square =0;

for j=1:a2j
	for k=2:a2k
		Theta1Square= Theta1Square + (Theta1(j,k)*Theta1(j,k));
	end
end



for j=1:a3j
	for k=2:a3k
		Theta2Square= Theta2Square + (Theta2(j,k)*Theta2(j,k));
	end
end


RegWeight = (lambda/(2*m)) * (Theta1Square+Theta2Square); 

J=J+RegWeight;




%%% Calculation of back propagation.

D2=zeros(size(Theta2));
D1=zeros(size(Theta1));

y = eye(num_labels)(y,:);



for t=1 : m 
	a1=X(t,:)';
	size(a1);
	
 	a1=[1;a1];  % Add bias term
	size(a1);

	z2=Theta1 * a1;
	a2=sigmoid(z2);
	
	a2=[1;a2];
	z3=Theta2 * a2;
	a3=sigmoid(z3);
	
	
	
	%% back propagation
	
		delta3 = a3-y(t,:)';
	    delta2 = ( (Theta2') * delta3 ) .* sigmoidGradient([1; z2]);
  	
		D2=D2+delta3 * a2';
	
		D1=D1+delta2(2:length(delta2)) * a1';
	 
end

Theta1_grad=D1/m;
Theta2_grad=D2/m;


%Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda/m) * Theta1(:,2:end)); 
%Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ((lambda/m) * Theta2(:,2:end));



Theta1_grad = Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];

Theta2_grad = Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
