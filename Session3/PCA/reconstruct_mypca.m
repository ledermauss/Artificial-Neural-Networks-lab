function [Xhat, error] = reconstruct_mypca(reduced, Et, original)
% expect columns as data points
Xhat = Et' * reduced;  
error = sqrt(mean(mean((original - Xhat).^2)));
end