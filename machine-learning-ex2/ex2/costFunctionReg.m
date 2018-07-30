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


%Regularized Cost Function
    [unregularized_J,unregularized_grad] = costFunction(theta,X,y);
    J = regularizedTheta(unregularized_J,theta, lambda);
    grad = regularizedGradient(unregularized_grad, theta,lambda);
    
    function J = regularizedTheta(unregularized_J,theta, lambda) 
        regularization_term=sum(theta([2:size(theta)]).^2)*(lambda/(2*m));
        J = unregularized_J + regularization_term;
    end

    function grad = regularizedGradient(unregularized_grad,theta,lambda)
        grad = unregularized_grad;
        theta_for_regularization = theta(2:size(theta));
        regularization_term = (lambda/m)*theta_for_regularization;
        grad(2:size(unregularized_grad)) = grad(2:size(unregularized_grad))+regularization_term;
    end
% =============================================================

    
end
