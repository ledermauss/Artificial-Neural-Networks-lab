%% 2. FeedFordward Networks
%% Exercice 1
% _Take y = sin(x^2) for 0:0.5:3*pi as a simple non linear function. Approximate 
% it with a NN using 1 hidden layer. use different algorithms. _
% 
% _How does gradient descent perforrm compared to other training algorithm? 
% Use *algorlm1* from Toledo. The script compares Levenberg-Marquardt "tarinlm" 
% with quasi-Newton "trainbfg" algorihtms_
%%
x = 0:0.05:3*pi;
y = sin(x.^2);


%% Choosing the best algorithm in terms of performance
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
NNlm = NN1Hidden(80, 'trainlm', 1000, x, y, false, false);
NNgd = NN1Hidden(80, 'traingd', 1000, x, y, false, false);

%% Testing neurons
% Here, the goal is to obviate the epochs, see the best they can do
%% Whole data
% V is NHidden x 1, W = 1 x NHidden
% here: W (V * x(NHidden*1)) (1*1)

sim_lm = NNlm.simulateData();
sim_gd = NNgd.simulateData();
figure
subplot(1,2,1)
plot(x, y, "b", x, sim_lm, "r");
title('lm - whole');
legend('target','predicted','Location','north');
subplot(1,2,2)
plot(x, y, "b", x, sim_gd, "r");
title('gd - whole');
legend('target','predicted','Location','north');
%% Train, test, regression
%%

sim_lm = NNlm.simulateData();
sim_gd = NNgd.simulateData();
[trainSet_lm, trainSim_lm] = NNlm.simulateTrain();
[trainSet_gd, trainSim_gd] = NNgd.simulateTrain();
[testSet_lm, testSim_lm] = NNlm.simulateTest();
[testSet_gd, testSim_gd] = NNgd.simulateTest();

figure
subplot(2,,1)
plot(trainSet_lm, NNlm.ytrain, "bx", trainSet_lm, trainSim_lm, "r");
title('lm - train set');
legend('target','predicted','Location','north');
subplot(2,3,2)
plot(testSet_lm, NNlm.ytest, "b",  testSet_lm, testSim_lm, "r");
title('lm - test set');
legend('target','predicted','Location','north');
subplot(2,3,3)
%postreg(testSim_lm, NNlm.ytest)
subplot(2,3,4)
plot(trainSet_gd, NNgd.ytrain, "bx", trainSet_gd, trainSim_gd, "r");
title('gd - train set');
legend('target','predicted','Location','north');
subplot(2,3,5)
plot(testSet_gd, NNgd.ytest, "b",  testSet_gd, testSim_gd, "r");
title('gd - test set');
legend('target','predicted','Location','north');
subplot(2,3,6)
%postreg(testSim_gd, NNgd.ytest)

%% Train, test, regression
figure
subplot(2,2,1)
[~, ~, R5] = NN5Neurons.testRegression();
subplot(2,2,2)
[~, ~, R10] = NN10Neurons.testRegression();
subplot(2,2,3)
[~, ~, R20] = NN20Neurons.testRegression();
subplot(2,2,4)
[~, ~, R40] = NN40Neurons.testRegression();
trainMSE = [NN5Neurons.trainMSE; NN10Neurons.trainMSE;
    NN20Neurons.trainMSE; NN40Neurons.trainMSE];
testMSE = [NN5Neurons.testMSE; NN10Neurons.testMSE;
    NN20Neurons.testMSE; NN40Neurons.testMSE];
R = [R5 R10 R20 R40]'
Epochs = [NN5Neurons.res.num_epochs;
    NN10Neurons.res.num_epochs; NN20Neurons.res.num_epochs;
    NN40Neurons.res.num_epochs]

Results =[R, testMSE, trainMSE, Epochs]
% TODO: add time column (absolute or per epoch)
%% 
%% Testing with under and overfitting
% Lets see how train and test behave with no validation, with over and underfitting
%%
NN100Neurons = NN1Hidden(100, 'trainlm', 1000, x, y, true);
NN5Neurons = NN1Hidden(5, 'trainlm', 1000, x , y, true);
res100 = NN100Neurons.res;
res5 = NN5Neurons.res;
plot(res100.epoch, res100.perf, "b", ...
    res100.epoch, res100.tperf, "r");
title('100 neurons Overfit');
legend('train','test','Location','north');

plot(res5.epoch, res5.perf, "b", ...
    res5.epoch, res5.tperf, "r");
title('5 neurons Underfit');
legend('train','test','Location','north');

%% some shit to force only training on the net.
%%
%should only work with trainParam.trainRatio = 1
% I believe it is crap
A = [x' y'];
k = round(length(x)*0.7);
[~, idx] = datasample(A, k, 'Replace',false);
trainData = A(idx);
testData = A(setdiff(idx, 1:length(x))); % to choose unselected
NN100Neurons = NN1Hidden(100, 'trainlm', 100,...
    trainData(:,1), trainData(:,2), true);
NN5Neurons = NN1Hidden(5, 'trainlm', 100,...
    trainData(:,1), trainData(:,2), true);
res100 = NN100Neurons.res;
res5 = NN5Neurons.res;
plot(res100.epoch, res100.perf, "b", ...
    res100.epoch, res100.tperf, "r");
title('100 neurons Overfit');
legend('train','test','Location','north');

plot(res5.epoch, res5.perf, "b", ...
    res5.epoch, res5.tperf, "r");
title('5 neurons Underfit');
legend('train','test','Location','north');
%% 
% Dudas (resueltas): 
% 
%  dónde medimos el error para comparar los modelos?  En el test (tperf)
% 
% * Valor de convergencia para test y train? La NN minimiza el error en todo. 
% Como hay un validation, evita que haya overfit y el test error suba mientras
% * el train sigue bajando.
% * I could check for large number of neurons trainMSE vs testMSE to see overfitting. 
% Same way around for underfit. (already in heatmap)
% * Puedo ajustar la forma de dividir train, validation y test usando net.divideParams.testRatio/trainRatio/validationRatio. 
% Si pongo trainRatio = 1, puedo ver cómo nunca converge y el train baja y baja.
% 
%