%% Redundancy and Random data
% Nore: my function expects data as columns, and returns the data as columns

q = 10;
gauss = randn(50,500);
[gEt, greduced, geigvals] = mypca(gauss, q);
[COEFF, SCORE] = pca(gauss', 'NumComponents', q);
% SCORE sare the projected values. Rows = observations
% Coeff corresponds to G (transformation matrix)
[ghat, gerror] = reconstruct_mypca (greduced, gEt, gauss)

%%

g_evol = plot_q(gauss, 5, 50, 5);
%%
load choles_all
[pEt, preduced, peigvals] = mypca(p, 2);
[phat, perror] = reconstruct_mypca (preduced, pEt, p);

p_evol = plot_q(p, 1, 21, 1);
%% 
% Comparing with the matlab implementation
% This is broken
     
[X,  mapstdSettings] = mapstd(p);
[Y, PCAsettings] = processpca(X, 'maxfrac',0.01);
% maxfrac: variance threshold for keeping components
Xhat = processpca('reverse', Y, PCAsettings);
phat2 = mapstd('reverse', Xhat, mapstdSettings);

perror2 = mean_squared_error(phat2, p)
%% 
% Going back to the original data using mapstd returns an error with the 
% same order of magnitud regarding p. Although I find the error between phat and 
% phat2 still too big. Anyway, the dimensions match: with a maxfrac of 0.01, only 
% the first two components are chosen. Those are the same I get with var > 0.01
%%
function [q_table] = plot_q(data, q_start, q_end, q_step)
    q_table = zeros((q_end - q_start)/q_step + 1, 3);
    i = 0;
    for q = q_start:q_step:q_end
        i = i + 1;
        [Et, reduced, eigvals] = mypca(data, q);
        [~, error] = reconstruct_mypca(reduced, Et, data);
        q_table(i, :) = [q, error, sum(eigvals)];
    end
    subplot(1,2,1)
    plot(q_table(:,1), q_table(:,2), 'r');
    legend('error',  'Location', 'south')
    subplot(1,2,2)
    plot(q_table(:,1), q_table(:,3), 'b');
    legend('cum_variance', 'Location', 'south')
    xlabel('q dimensions')
end

function [error] = mean_squared_error(A, B)
    error = sqrt(mean(mean((A - B).^2)));
end