%% eUSPS hand written digits
%% Mean three

% clc
% clear all
% close all
load threes -ascii
threes = threes'; % cols as observations
print_digit(threes, 99);
mean_three = mean(threes, 2); % mean of rows (feats)
print_digit(mean_three, 0)
%% Scaled Eigenvalues
%%
[Et, reduced, eigvals] = mypca(threes', 256);
figure;
plot(eigvals)
title("Eigenvalues scaled")
%% 4 first PC
%%
% don't forget checking if observations are cols!
% Setup
samples = [100, 101, 102, 200, 250, 300 , 350, 400, 450, 500];
i = 1;
PC = [1, 2, 3, 4, 16, 64, 128, 256];
[ha, pos] = tight_subplot(length(PC) + 1,1, 0,[.01 .01],[.05 .01]);

% Original numbers
original = plot_mosaic(threes, 1, length(samples), samples)
axes(ha(i));
colormap('bone');
imagesc(1- original, [0,1]);
set(gca,'XTick',[], 'Ytick', [], 'FontSize', 18); % remove X and Y labels
ylabel('Original');
% reconstructed data
for q=PC
    reconstruct = compress_reconstruct(threes, q);
    % figure; hold on; % to plot many times
    mosaic = plot_mosaic(reconstruct, 1, length(samples), samples);
     
    i = i + 1;
    axes(ha(i));
    colormap('bone');
    imagesc(1- mosaic, [0,1]);
    set(gca,'XTick',[], 'Ytick', [], 'FontSize', 18); % remove X and Y labels
    ylabel([num2str(q)]);

end
%set(ha(2:length(PC),'XTickLabel','')
    
%% 
% Duds: porqué el primer componente está negro? O sea, hay valores muy pequeños, 
% no tiene sentido si el primer componente es tan grande

one_dim = compress_reconstruct(threes, 1)
%% Error - variance plot
% Also compared to random data
%%
figure;
q_table = error_by_pc(threes,1,251,10)
title('Digits')
gauss = randn(50,500);
g_evol = error_by_pc(gauss, 5, 50, 5);
title('random data')

%% 
% TODO:
% 
% * Analyze components like zotero paper: what is the first component, what 
% is the second one (component, not sum)
%%

%%
function [] = print_digit(digits, i)
    % cols as observations
    colormap('bone')
    if i == 0
        imagesc(reshape(1 - digits, [16, 16]), [0,1])
    else
        imagesc(reshape(1- digits(:,i), [16, 16]), [0,1])
    end
end

function [reconstruc] = compress_reconstruct(data, q)
    % data as cols
    [Et, reduced, ~] = mypca(data, q);
    [reconstruc, ~] = reconstruct_mypca(reduced, Et, data);
end

function [mosaic] = plot_mosaic(data, rows, cols, sample_cols)
    if length(sample_cols) ~= rows*cols
        % rows*cols == length(sample_cols), otherwise error
        error("Dimension mismatch; too many/few smaples")
    end
    % data as cols
    sampled_data = data(:, sample_cols);
    % reshape columnwise, only 1 row by now
    mosaic = reshape(sampled_data, [16, cols*16]);
end