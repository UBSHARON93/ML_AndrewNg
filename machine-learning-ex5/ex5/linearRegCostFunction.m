function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% calculate the hypothesis
h = X*theta;
% Hypothesis error
h_error = h - y;
% sum of squared error
sumSquaredError = sum(h_error .^ 2);
% Now first term in Cost fcn 
Reg_LinR_FirstTerm = (1/(2 * m)) * sumSquaredError;
% calculate second term
theta_without_Bias = theta(2:end);
sum_SquaredTheta = sum(theta_without_Bias .^ 2);
Reg_LinR_SecondTerm = (lambda / (2 * m)) * sum_SquaredTheta;

% Cost function eqtn
J = Reg_LinR_FirstTerm + Reg_LinR_SecondTerm;

% Calculate gradients
grad = (1/m) * X' * h_error;
grad(2:end) += (lambda / m) * theta_without_Bias;
% =========================================================================

grad = grad(:);

end
