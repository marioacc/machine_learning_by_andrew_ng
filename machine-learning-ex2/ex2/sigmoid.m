function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
numerator = 1;
denominator = 1 + e.^(-z);
sigmoid_value = numerator./denominator;
g = sigmoid_value;

% =============================================================

end
