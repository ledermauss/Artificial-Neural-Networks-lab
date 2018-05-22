%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% USPS hand written digits %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Mean three

% clc
% clear all
% close all
load threes -ascii
threes = threes'; % cols as observations
print_digit(threes, 99);
mean_three = mean(threes, 2); % mean of rows (feats)
print_digit(mean_three)
set(gca,'XTick',[], 'Ytick', [], 'FontSize', 18, 'Visible', 'off'); % remove X and Y labels

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Componentwise reconstruction %%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% setup 
% samples = [100, 101, 102, 200, 250 , 300, 350, 400, 450, 500]; % size 10
samples = [100, 101, 102, 200];
i = 1;
PC = [1 16 64 128 256];
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
% Extra ideas
% - Plot the components themselves
% - Plot for just one three the reconstructions with [1 2 3 4 16 54 128
% 256] in one or two lines
%% One threee only
PC = [1 2 3 4 16 64 128 256];
% removing the margins
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
% original Digit
print_digit(threes, 102);
set(gca,'XTick',[], 'Ytick', [], 'Visible', 'off') % remove X and Y labels
print(gcf,'images/single_three/original','-dpng')
% by components
for q=PC
    reconstruct = compress_reconstruct(threes, q);
    colormap('bone');
    print_digit(reconstruct, 102);
    set(gca,'XTick',[], 'Ytick', [], 'Visible', 'off'),  % remove X and Y labels
    print(gcf,['images/single_three/PC' num2str(q)],'-dpng')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%       Error and Variance     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Scaled Eigenvalues

[Et, reduced, eigvals] = mypca(threes', 256);
figure;
plot(eigvals)
title("Eigenvalues scaled")
%%
figure;
q_table = error_by_pc(threes,1,251,10);
title('Digits')
% Also compared to random data
gauss = randn(50,500);
g_evol = error_by_pc(gauss, 5, 50, 5);
title('random data')

%% 
% TODO:
% 
% * Analyze components like zotero paper: what is the first component, what 
% is the second one (component, not sum)
%%


function [] = print_digit(dgt, i)
    % cols as observations
    colormap('bone')
    if nargin < 2  % just one digit provided
        imagesc(reshape(1 - dgt, [16, 16]), [0,1])
    else
        imagesc(reshape(1- dgt(:,i), [16, 16]), [0,1])
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