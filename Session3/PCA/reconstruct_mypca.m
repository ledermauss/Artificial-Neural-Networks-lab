function [Xhat, error] = reconstruct_mypca(reduced, Et, original)
% expect columns as data points
% -----  Output ------
% Xhat: reoconstructed original data (d * N)
% error: MSE(X - Xhat)
Xhat = (Et' * reduced) + mean(original, 2);  
% error = sqrt(mean(mean((original - Xhat).^2)));
error = mean(sum((original - Xhat).^2));
%error = mse(original-  Xhat);
% should actually be (mean(sum(.))) no sqrt
end
