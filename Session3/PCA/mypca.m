function [Et, reducedX, eigvals, eig_trace] = mypca(data, q, scale)  
% q: desired dimensions
    if nargin < 3
        scale = true;
    end
    x = data';                  % observation as rows (expected as columns)
    [eigvec, eigvals] = eigs(cov(x), q);  %cov centers matrix automatically
    [~, all_eigvals] = eig(cov(x));  % easier to 
    if scale == true
        eigvals = diag(eigvals) /trace(all_eigvals);
    else
        eigvals = diag(eigvals);
    end
    Et = eigvec';                   % eigvecs: p (original dims) * q
    reducedX = Et * (data - mean(data,2));
    eig_trace = trace(all_eigvals);
end
 