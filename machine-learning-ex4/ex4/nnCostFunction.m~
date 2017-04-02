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
                 hidden_layer_size, (input_layer_size + 1)); % 25 * 401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); % 10 * 26

% Setup some useful variables
m = size(X, 1); % x rows
         
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


X = [ones(m,1) X]; % add bias term 1 in all inputs, 5000 * 401
a1 = X'; % 401 * 5000
z2 = Theta1 * a1; % 25 * 5000
a2 = sigmoid(z2); % 25 * 5000
a2 = [ones(1, m); a2] ; % add bias term 1, 26 * 5000
z3 = Theta2 * a2; % 10 * 5000
a3 = sigmoid(z3); % 10 * 5000, don't add bias term, as this is the output h

h = a3 ;

% convert each y output corresponding to a training example into a vector
% y is 5000 * 1, should be 10 * 5000
c = [1:1:num_labels]'; % all classes in the output
% compute modified y, which is 10 * 5000, each column contains 1 if the corresponding input was 1
% every output is now a 10 dim column vector in the ymod matrix
% y' is 1 * 5000
ymod = c == y'(1,1:m); % 10 * 5000

% ymod' is 5000 * 10

% vectorized summation over all K
 sum_all_k = -1 * ymod' * log(h) - (1-ymod') * log(1-h); % 5000 * 5000
 
diag_sum_all_k = diag(sum_all_k); % 5000 * 1
sum_all_training_sets_m = sum(diag_sum_all_k);

J = (1/m)  * sum_all_training_sets_m; % non regularized

% regularization term is nothing but the sum of squares of thetas in all layers excluding the theta weights for the bias units whose value is always 1
J = J + (lambda/(2*m)) * ( sum((Theta1 .* Theta1)(:,2:end)(:)) + sum((Theta2.*Theta2)(:,2:end)(:)));

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
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
%Delta1 Theta1 Theta1_grad are same size 
%Delta2 Theta2 Theta2_grad are same size 
for t = 1:m
    %Step1, feed forward pass, iterate over each training example
    a1 = X'(:,t); % 401 * 1
    z2 = Theta1 * a1; % 25 * 1
    a2 = sigmoid(z2); % 25 * 1
    a2 = [1; a2] ; % add bias term = 1, 26 * 1
    z3 = Theta2 * a2; % 10 * 1
    a3 = sigmoid(z3); % 10 * 1, output reached, don't add bias term 
    ymod = c == y'(1,t); % 10 * 1
    % Step 2
    delta_3 = a3 - ymod; % 10 * 1
    % Step 3
    delta_2 = (Theta2' * delta_3) .* a2 .* (1-a2); % 26 * 1
%    delta_2 = (Theta2(:,2:end)' * delta_3) .* a2(2:end,:) .* (1-a2(2:end,:)); % 25 * 1
    delta_2 = delta_2(2:end); % 25 * 1, removing delta for bias term
    % Step 4
%    Theta1_grad = Theta1_grad + delta_2 * a1'; % 25 * 401
%    Theta2_grad = Theta2_grad + delta_3 * a2'; % 10 * 26
    Delta1 = Delta1 + delta_2 * a1'; % 25 * 401
    Delta2 = Delta2 + delta_3 * a2'; % 10 * 26
endfor

Theta1_grad = Delta1/m;
Theta2_grad = Delta2/m;
grad = [Theta1_grad(:) ; Theta2_grad(:); ];


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad = [Theta1_grad(:,1) Theta1_grad(:,2:end) + (lambda/m) .* Theta1(:,2:end)];

Theta2_grad = [Theta2_grad(:,1) Theta2_grad(:,2:end) + (lambda/m) .* Theta2(:,2:end)];
grad = [Theta1_grad(:) ; Theta2_grad(:); ];
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
