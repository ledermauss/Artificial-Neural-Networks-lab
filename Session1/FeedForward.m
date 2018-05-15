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
neurons = [5, 10, 20, 40, 80];
experiments = 20;
algs = {'traingd', 'traingda', 'traincgf', 'traincgp', 'trainbfg' 'trainlm'}; 
algs = {'trainlm'};
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
NN5Neurons = NN1Hidden(5, alg, 1000, x, y, false, true);
%% 
% DAVID: compara dos algoritmos en el segundo plot
% : in the bayesian part, show that there is no validation (and thus it is
% slower)
NN10Neurons = NN1Hidden(10, alg, 1000, x, y, false, false);
NN20Neurons = NN1Hidden(20, alg, 1000, x, y, false, false);
NN40Neurons = NN1Hidden(80, alg, 1000, x, y, false, true);

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
Epochs = [NN5Neurons.num_epochs;
    NN10Neurons.num_epochs; NN20Neurons.num_epochs;
    NN40Neurons.num_epochs]

Results = [R, testMSE, trainMSE, Epochs]
% TODO: add time column (absolute or per epoch)

%% Testing neurons
% Here, the goal is to obviate the epochs, see the best they can do
%%
% V is NHidden x 1, W = 1 x NHidden
% here: W (V * x(NHidden*1)) (1*1)


y_test5 = NN5Neurons.simulateData();
y_test10 = NN10Neurons.simulateData();
y_test20 = NN20Neurons.simulateData();
y_test40 = NN40Neurons.simulateData();
figure
subplot(2,2,1)
plot(x, y, "b", x, y_test5, "r");
title('5 neurons');
legend('target',alg,'Location','north');
subplot(2,2,2)
plot(x, y, "b", x, y_test10, "r");
title('10 neurons');
legend('target',alg,'Location','north');
subplot(2,2,3)
plot(x, y, "b", x, y_test20, "r");
title('20 neurons');
legend('target',alg,'Location','north');
subplot(2,2,4)
plot(x, y, "b", x, y_test40, "r");
title('40 neurons');
legend('target',alg,'Location','north');


%train_mask = res.trainMask;
% train_mask{1} returns a vector with 1 for the chosen data
%%
[test5, y_test5] = NN5Neurons.simulateTest();
[test10, y_test10] = NN10Neurons.simulateTest();
[test20, y_test20] = NN20Neurons.simulateTest();
[test40, y_test40] = NN40Neurons.simulateTest();

figure
subplot(2,2,1)
plot(x, y, "b", x, y_test5, "r");
title('5 neurons');
legend('target',alg,'Location','north');
subplot(2,2,2)
plot(x, y, "b", x, y_test10, "r");
title('10 neurons');
legend('target',alg,'Location','north');
subplot(2,2,3)
plot(x, y, "b", x, y_test20, "r");
title('20 neurons');
legend('target',alg,'Location','north');
subplot(2,2,4)
plot(x, y, "b", x, y_test40, "r");
title('40 neurons');
legend('target',alg,'Location','north');
%% 
% Now, plot the regressions
%%
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