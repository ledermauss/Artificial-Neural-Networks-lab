%% Bayesian Inference

x = 0:0.05:3*pi
y = sin(x.^2)
%%
neurons = [50, 100, 150, 300];
algs = {'trainlm', 'trainbr'};

experiments = 10;
res = zeros(length(neurons), 6, length(algs));
i = 1;
for alg= algs
    algResults = [];
    for nhidden = neurons
        neuronExps = zeros(experiments, 5);
        for e=1:experiments
            e
            % 1000 iterations, let it converge
            NN = NN1Hidden(nhidden, alg{1}, 150, x, y, false, false);
            [~, ~, R] = NN.testRegression();
            neuronExps(e,:) = [R NN.testMSE NN.trainMSE NN.epochs NN.time];
        end
        avgExps = mean(neuronExps);
        algResults = [algResults; nhidden  avgExps]; % length(neurons) * 6 array
    end
    res(:,:,i) = algResults;
    i = i + 1
end
