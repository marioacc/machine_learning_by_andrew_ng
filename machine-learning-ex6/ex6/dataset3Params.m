function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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
values_to_try = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
errors = zeros(length(values_to_try),length(values_to_try));
for i = 1:length(values_to_try)
    C = values_to_try(i);
    for j = 1:length(values_to_try)
        sigma = values_to_try(j);
        model = svmTrain(X,y,C,@(x1,x2) gaussianKernel(x1,x2,sigma));
        predictions = svmPredict(model,Xval);
        error =  mean(double(predictions ~= yval));
        errors(i,j) = error;
    end
end
[index_for_C,index_for_sigma]= find(errors == min(min(errors)));

C = values_to_try(index_for_C);
sigma = values_to_try(index_for_sigma);
% =========================================================================

end
