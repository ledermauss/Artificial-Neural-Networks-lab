function [Et, reducedX, scaled_eigvals] = mypca(data, q)   % q: desired dimensions
    x = data';                  % observation as rows (expected as columns)
    [eigvec, eigvals] = eigs(cov(x), q);  %cov centers matrix automatically
    [~, all_eigvals] = eig(cov(x));  % easier to 
    scaled_eigvals = diag(eigvals) /trace(all_eigvals);
    Et = eigvec';                   % eigvecs: p (original dims) * q
    reducedX = Et * data;
end
 