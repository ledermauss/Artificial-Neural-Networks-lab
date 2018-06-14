wine = dlmread('winequality_data/winequality-white.csv', ';');
positive = wine(:,12)==5;
negative = wine(:,12)==6;
wine_neg = wine(negative, 1:11);
wine_pos = wine(positive, 1:11);
%%
X = [wine_neg;wine_pos]';
%one hot encoding
Y = [ones(1,size(wine_neg,1)) zeros(1,size(wine_pos, 1)); 
    zeros(1,size(wine_neg,1)) ones(1,size(wine_pos,1))];

%%
% https://nl.mathworks.com/help/nnet/pattern-recognition-and-classification.html
Rvaz = NN1Pattern(10, 'trainrp', 1000, X, Y, 'logsig', true)



%%
% training algorithms: 
% https://nl.mathworks.com/help/nnet/ug/train-and-apply-multilayer-neural-networks.html#bss331l-2
fcnTransfer = {'logsig', 'tansig', 'purelin', 'softmax'}
trainAlgs = {'trainscg', 'trainrp', 'traingd'}%, 'trainlm', 'trainbfg'} 
experiments = 20;
neurons = [5 10 20 40 60 80 100]
i = 1;
res = zeros(length(neurons), 4, length(trainAlgs));
%for transfer= fcnTransfer
for trainAlg = trainAlgs
    algResults = zeros(length(neurons), 4);
    n = 1;
    for nhidden = neurons
        neuronExps = zeros(experiments, 3);
        for e=1:experiments
            % 1000 iterations, let it converge
            NN = NN1Pattern(nhidden, trainAlg{1}, 500, X, Y, 'logsig', false);
            valCCR = NN.valCCR();
            neuronExps(e,:) = [valCCR NN.epochs NN.time];
        end
        avgExps = mean(neuronExps,1);
        algResults(n,:) = [nhidden  avgExps]  
        n = n + 1;
    end
    res(:,:,i) = algResults;
    i = i + 1;
end

%%%%%%%%%%%%%%%%%%%%%%%
%%  Components
%%%%%%%%%%%%%%%%%%%%%%%
[Et, reduced, eigvals] = mypca(X, 11);
figure;
bar(eigvals);
title("Eigenvalues scaled")
set(gca,'FontSize', 14);
%%
error_by_pc(X,1,11,1, true);
title('Reconstruction MSE')
set(gca,'FontSize', 16);
%% reconstruct with 3 dimensions
[Et, Xs, eigvals] = mypca(X, 3);
gplotmatrix(Xs', [], Y(1,:)')

%%%%%%%%%%%%%%%%%%%%
%% Classify on Xs %%
fcnTransfer = {'logsig', 'tansig', 'purelin', 'softmax'}
trainAlgs = {'trainscg', 'trainrp', 'traingd'}%, 'trainlm', 'trainbfg'} 
experiments = 20;
neurons = [5 10 20 40 60 80]
i = 1;
res = zeros(length(neurons), 4, length(fcnTransfer));
%for transfer= fcnTransfer
for transferFcn = fcnTransfer
    algResults = zeros(length(neurons), 4);
    n = 1;
    for nhidden = neurons
        neuronExps = zeros(experiments, 3);
        for e=1:experiments
            % 1000 iterations, let it converge
            NN = NN1Pattern([nhidden*3 nhidden*2 nhidden] , 'trainrp', 500, Xs, Y, transferFcn{1}, false);
            valCCR = NN.valCCR();
            neuronExps(e,:) = [valCCR NN.epochs NN.time];
        end
        avgExps = mean(neuronExps,1);
        algResults(n,:) = [nhidden  avgExps]  
        n = n + 1
    end
    res(:,:,i) = algResults;
    i = i + 1
end

