%% 2. FeedFordward Networks
%% Exercice 1
% _Take y = sin(x^2) for 0:0.5:3*pi as a simple non linear function. Approximate 
% it with a NN using 1 hidden layer. use different algorithms. _
% 
% _How does gradient descent perforrm compared to other training algorithm? 
% Use *algorlm1* from Toledo. The script compares Levenberg-Marquardt "tarinlm" 
% with quasi-Newton "trainbfg" algorihtms_
%%
clear all
x = 0:0.05:3*pi;
y = sin(x.^2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Choosing the best algorithm in terms of performance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% setup
%neurons = [5, 10, 20, 40, 80, 100];
neurons = [100, 120];

experiments = 20;
algs = {'traingd', 'traingda', 'traincgf', 'traincgp', 'trainbfg' 'trainlm'}; 
res = zeros(length(neurons), 6, length(algs));

i = 1;
for alg= algs
    algResults = [];
    for nhidden = neurons
        neuronExps = zeros(experiments, 5);
        for e=1:experiments
            % 1000 iterations, let it converge
            NN = NN1Hidden(nhidden, alg{1}, 1000, x, y, false, false);
            [~, ~, R] = NN.testRegression();
            neuronExps(e,:) = [R NN.testMSE NN.trainMSE NN.epochs NN.time];
        end
        avgExps = mean(neuronExps);
        algResults = [algResults; nhidden  avgExps]; % length(neurons) * 6 array
    end
    res(:,:,i) = algResults;
    i = i + 1;
end
%%
%% Optimal configs, gd and lm
% DAVID: compara dos algoritmos en el segundo plot
% : in the bayesian part, show that there is no validation (and thus it is
% slower)
% V is NHidden x 1, W = 1 x NHidden
% here: W (V * x(NHidden*1)) (1*1)
NNlm = NN1Hidden(80, 'trainlm', 1000, x, y, false, false);
NNgd = NN1Hidden(80, 'traingd', 1000, x, y, false, false);

%% Whole data and regression
%%

sim_lm = NNlm.simulateData();
sim_gd = NNgd.simulateData();
[trainSet_lm, trainSim_lm] = NNlm.simulateTrain();
[trainSet_gd, trainSim_gd] = NNgd.simulateTrain();
[testSet_lm, testSim_lm] = NNlm.simulateTest();
[testSet_gd, testSim_gd] = NNgd.simulateTest();

% % Train
% plot(trainSet_lm, NNlm.ytrain, "bx", trainSet_lm, trainSim_lm, "r");
% title('lm - train set');
% legend('target','predicted','Location','north');
% % Test
% plot(testSet_lm, NNlm.ytest, "b",  testSet_lm, testSim_lm, "r");
% title('lm - test set');
% legend('target','predicted','Location','north');
% Test regression

plot(x, y, "b", x, sim_lm, "r");
 set(gca, 'FontSize', 12);
title('Levenberg - Marquardt');
legend('target','predicted','Location','north'), 
fprintf(" Press a key to continue"), pause

plotregression(testSim_lm, NNlm.ytest, 'Levenberg Marquardt - Test Regression'),
 set(gca, 'FontSize', 12),
fprintf(" Press a key to continue"), pause
close
% % Train
% plot(trainSet_gd, NNgd.ytrain, "bx", trainSet_gd, trainSim_gd, "r");
% title('gd - train set');
% legend('target','predicted','Location','north');
% % Test
% plot(testSet_gd, NNgd.ytest, "b",  testSet_gd, testSim_gd, "r");
% title('gd - test set');
% legend('target','predicted','Location','north');
figure;
plot(x, y, "b", x, sim_gd, "r"), set(gca, 'FontSize', 12),
title('Gradient Descent (batch)');
legend('target','predicted','Location','north'), 
fprintf(" Press a key to continue"), pause

plotregression(testSim_gd, NNgd.ytest, 'Gradient Descent - Test Regression'),
set(gca, 'FontSize', 12),
fprintf(" Press a key to continue"), pause
close

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Testing with under and overfitting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lets see how train and test behave with no validation, with over and underfitting
%% overfit
NN1Hidden(100, 'trainlm', 1000, x, y, true, true),pause % no val set (overfitting)
NN1Hidden(100, 'trainlm', 1000, x, y, false, true),pause % val set (stop overfit)
%% underfit
% 300 iter is long enough
NN1Hidden(5, 'trainlm', 300, x, y, true, true),pause % no val set (underfit)
NN1Hidden(5, 'trainlm', 300, x, y, false, true),pause % val set (stop underfit)

%% perfect fit
NN1Hidden(80, 'trainlm', 1000, x, y, true, true),pause % no val set (underfit)
NN1Hidden(80, 'trainlm', 1000, x, y, false, true),pause % val set (stop underfit)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Gaussian  Noise        %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
x = 0:0.01:3*pi;
y = sin(x.^2);
noise = [0.2, 0.8 2];
noiseRes = [];
for bw=noise% set
    noise = randn(size(x)) .* bw;
    y_noise = y + noise;

    NN = NN1Hidden(80, 'trainlm', 1000, x, y_noise, false, false);
    sim_noise = NN.simulateData();
    [testSet_noise, testSim_noise] = NN.simulateTest();

    figure
    plot(x, y, "--b" ,x, y_noise, ":r",  x, sim_noise, "k"), % best config
    set(gca, 'FontSize', 12),
    title(['BW = ' num2str(bw) ', extra data'], 'FontSize', 18),
    legend('target', 'noise', 'predicted','Location','north'), pause,
    fprintf('press a key')

    [~, ~, R] = NN.testRegression();
    noiseRes = [ noiseRes; bw R NN.testMSE NN.trainMSE NN.epochs NN.time];
end



%%
% Dudas (resueltas): 
% % 
% * Valor de convergencia para test y train? La NN minimiza el error en todo. 
% Como hay un validation, evita que haya overfit y el test error suba mientras
% * el train sigue bajando.
% * I could check for large number of neurons trainMSE vs testMSE to see overfitting. 
% Same way around for underfit. (already in heatmap)
% * Puedo ajustar la forma de dividir train, validation y test usando net.divideParams.testRatio/trainRatio/validationRatio. 
% Si pongo trainRatio = 1, puedo ver cómo nunca converge y el train baja y baja.
% 
%