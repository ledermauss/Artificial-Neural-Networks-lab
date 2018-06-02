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
%set(gca,'XTick',[], 'Ytick', [], 'FontSize', 18, 'Visible', 'off'); % remove X and Y labels

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Componentwise reconstruction %%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Mosaic (many threes)
% setup 
% samples = [100, 101, 102, 200, 250 , 300, 350, 400, 450, 500]; % size 10
samples = [100, 101, 102, 200];
i = 1;
PC = [1 16 64 128 256];
% tight subplot: no spaces between plots
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

PC = [1 2 3 4 16 50 64 128 256];
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
bar(eigvals);
title("Eigenvalues scaled")
set(gca,'FontSize', 14); 

%% Digits dataset %%
%% Reconstruction error vs 1 - k_cumsum(var)
figure;
digits_error = error_by_pc(threes,1,256,5, true);
title('Reconstruction MSE vs eigenvalues variance (Digits)')
set(gca,'FontSize', 16);
%% Regression
% note: second col (y axis) is the MSE error, third the cumsum
% regression: first target (x axis), later output (y axis)
plotregression(digits_error(:,3), digits_error(:,2), 'Rec. Error - last eigvals fit'),
xlabel("Sum(k:end) eigvals"), ylabel("Reconstruction Error")
set(gca,'FontSize', 16);

%% Random data %%
%% ec error vs 1 - k_cumsum(var)
gauss = randn(256,500);
g_error = error_by_pc(gauss, 1, 256, 5, true);
title('Reconstruction MSE vs eigenvalues variance (Gaussians)')
set(gca,'FontSize', 16);

%% Regression
plotregression(g_error(:,3), g_error(:,2), 'Gaussians - Rec. Error fit'),
xlabel("Sum(k:end) eigvals"), ylabel("Reconstruction Error")
set(gca,'FontSize', 16);

%% Random trace vs digit trace
[~, ~, evals_threes] = mypca(threes', 50, false);
[~, ~, evals_gauss]  = mypca(gauss, 50, false);
plot(1:50, evals_gauss, 'g', 1:50, evals_threes, 'b')
legend("Gaussians eigenvalues", "Digits eigenvalues", 'Location', 'north'),
xlabel('Components'), ylabel('Eigenvalues (variance)'),
title("Digits vs Gaussians trace"),
set(gca, 'FontSize', 16)

%% Digits vs random rec MSE
plot(g_error(:,1), g_error(:,3)./g_error(1,3), 'g', ...
digits_error(:,1), digits_error(:,3)/digits_error(1,3), 'b'),
legend(' Gaussians MSE', 'Digits MSE', 'Location', 'north')
xlabel('Selected Components'),
title('Digits rec. MSE vs Gaussians rec. MSE (scaled)')
set(gca,'FontSize', 16),ylim([0 1])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Component meaning           %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% * Analyze components like zotero paper: what is the first component, what 
% is the second one (component, not sum). Two images
% * Scholkopf: plot the component themself
% * Componentwise plot from EPFL package
%% Eigvec plot
[Et, reducedX, eigvals] = mypca(threes, 256, false);
 %Et = pca(threes'); Et = Et';
 
[ha, pos] = tight_subplot(2, 3, [0.07 0.01],[.01 .07],[.01 .01]);
for i=1:6
    axes(ha(i))   
    print_digit(Et' * 3 , i), title(['PC' num2str(i)])  % * 3 just scales
    set(gca,'XTick',[], 'Ytick', [], 'FontSize', 13); % remove X and Y labels
end

%% Extract indexes of threes corresponding to PC quantiles
% values on those PC
comp1 = 1; comp2 = 5;
first_com = Et(comp1,:) * (threes - mean(threes,  2));
second_com = Et(comp2,:) * (threes - mean(threes, 2));
vals = [0.2 0.4 0.6 0.8 1];
% quantiles
first_quant = quantile(first_com, vals);
second_quant = quantile(second_com, vals);
first_col = first_com';
second_col = second_com';
indexes = zeros(length(vals)); % N*N
r = 0;
% finding the best matches
for q1=first_quant
    r = r + 1;
    c = 0;
    for q2=second_quant
        c = c + 1;
        % closest three (euclidean dist) to the quantile
        dists = sqrt((first_col - q1).^2 + (second_col - q2).^2);
        [val, index] = min(dists);
        indexes(r, c) = index;
    end
end

%% Plotting those threes
[ha, pos] = tight_subplot(length(vals), length(vals), 0,[.06 .01],[.05 .01]);
count = 0;
for i = 1:length(vals)
    for j = 1:length(vals)
        count = count + 1;
        axes(ha(count));
        %subplot(length(vals), length(vals), count)
        print_digit(threes, indexes(i, j))
        set(gca,'XTick',[], 'Ytick', [], 'FontSize', 16); % remove X and Y labels
        if j == 1 % labels on the first column
            ylabel([num2str(vals(i))])% ' PC' num2str(comp1)])
        end 
        if i == 5 % labels on the last row
            xlabel([num2str(vals(j))])% ' PC' num2str(comp1)])
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%
%% Cherrypicking good threes for the first component
% Just PC1 eigvec

[ha, pos] = tight_subplot(2, 1, [0.07 0.01],[.01 .07],[.01 .01]);

axes(ha(1))   
print_digit(Et' * 3, 1),    
set(gca,'XTick',[], 'Ytick', []), title('PC 1 ', 'FontSize', 16); % remove X and Y labels
axes(ha(2))
print_digit(Et' * 3, 5),  
set(gca,'XTick',[], 'Ytick', []), title('PC 5', 'FontSize', 16); % remove X and Y labels
%first_com = Et(comp,:) * (threes - mean(threes, 2),);
%vals = [0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
%quant = quantile(first_com, vals);

%% all the others
col = 3
for i=1:5
    print_digit(threes, indexes(i, col))
    set(gca,'XTick',[], 'Ytick', [], 'Visible', 'off'); % remove X and Y labels
    print(gcf,['images/PC_interpret/PC1_' num2str(i)],'-dpng');
end
%%%%%%%%%%%%%%%
%% Functions %%
%%%%%%%%%%%%%%%
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