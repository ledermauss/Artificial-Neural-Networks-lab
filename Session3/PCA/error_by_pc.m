function [pc_table] = error_by_pc(data, pc_start, pc_end, pc_step, inv_sum)
% create a table with the MSE reconstrunction error by included
% components. For every step, reduce dimensionality, then go back
% to the original one, and obtain the error
% ------- Inputs -----
% data: (d * N) data points
% inv_sum: boolean. If true, third column = 1- cumsum(variance) ( = variance
% of all but k first elements
% ------- Ouputs -----
% pc_table: table with [PC_total  rec_error  (sum|inv_sum]

if nargin < 5
    inv_sum = false;
end
    pc_table = zeros((pc_end - pc_start)/pc_step + 1, 3);
    i = 0;
    for q = pc_start:pc_step:pc_end
        i = i + 1;
        [Et, reduced, eigvals] = mypca(data, q);
        [~, error] = reconstruct_mypca(reduced, Et, data);
        if inv_sum == true
            pc_table(i, :) = [q, error,  1 - sum(eigvals)];
        else
            pc_table(i, :) = [q, error, sum(eigvals)];
        end
    end
    plot(pc_table(:,1), pc_table(:,2), 'r', ...
    pc_table(:,1), pc_table(:,3), 'b')
    legend('error', 'variance', 'Location', 'south')
    xlabel('q dimensions')
end

