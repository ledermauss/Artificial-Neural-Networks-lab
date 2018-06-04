clear all
close all
nntraintool('close');
nnet.guis.closeAllViews();

% Neural networks have weights randomly initialized before training.
% Therefore the results from training are different each time. To avoid
% this behavior, explicitly set the random number generator seed.
rng('default')


% Load the training data into memory
%[xTrainImages, tTrain] = digittrain_dataset;
load('digittrain_dataset');

%% Layer 1
hiddenSize1 = 100;
autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
    'MaxEpochs',150, ...  % few, but too slow otherwise
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

figure;
plotWeights(autoenc1);
feat1 = encode(autoenc1,xTrainImages);

%% Layer 2
hiddenSize2 = 81;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',120, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

feat2 = encode(autoenc2,feat1);
plotWeights(autoenc2);

%% Layer 3
hiddenSize3 = 64;
autoenc3 = trainAutoencoder(feat2,hiddenSize3, ...
    'MaxEpochs',120, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

plotWeights(autoenc3);
feat3 = encode(autoenc3,feat2);

%% Layer 4
hiddenSize4 = 49;
autoenc4 = trainAutoencoder(feat3,hiddenSize4, ...
    'MaxEpochs',120, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

plotWeights(autoenc4);
feat4 = encode(autoenc4,feat3);
%% Last layers
softnet2 = trainSoftmaxLayer(feat1,tTrain,'MaxEpochs',400);
softnet3 = trainSoftmaxLayer(feat2,tTrain,'MaxEpochs',400);
softnet4 = trainSoftmaxLayer(feat3,tTrain,'MaxEpochs',400);
softnet5 = trainSoftmaxLayer(feat4,tTrain,'MaxEpochs',400);


%% Deep Net
deepnet2 = stack(autoenc1,softnet2);
deepnet3 = stack(autoenc1,autoenc2,softnet3);
deepnet4 = stack(autoenc1,autoenc2, autoenc3, softnet4);
deepnet5 = stack(autoenc1,autoenc2, autoenc3, autoenc4, softnet5);


%% Test deep net
for deepnet = {deepnet2, deepnet3, deepnet4, deepnet5}
    imageWidth = 28;
    imageHeight = 28;
    inputSize = imageWidth*imageHeight;
    %[xTestImages, tTest] = digittest_dataset;
    load('digittest_dataset');
    xTest = zeros(inputSize,numel(xTestImages));
    for i = 1:numel(xTestImages)
        xTest(:,i) = xTestImages{i}(:);
    end
    y = deepnet{1}(xTest);
    figure;
    plotconfusion(tTest,y);
    classAcc=100*(1-confusion(tTest,y))
end

%% Test fine-tuned deep net
for deepnet = {deepnet5}
    xTrain = zeros(inputSize,numel(xTrainImages));
    for i = 1:numel(xTrainImages)
        xTrain(:,i) = xTrainImages{i}(:);
    end
    deepnet = train(deepnet{1},xTrain,tTrain);
    y = deepnet(xTest);
    figure;
    plotconfusion(tTest,y);
    classAcc=100*(1-confusion(tTest,y))
    view(deepnet)
end

%% Compare with normal neural network (1 hidden layers)
net = patternnet(100);
net=train(net,xTrain,tTrain);
y=net(xTest);
plotconfusion(tTest,y);
classAcc1=100*(1-confusion(tTest,y))
view(net)

% Compare with normal neural network (2 hidden layers)
net = patternnet([100 81]);
net=train(net,xTrain,tTrain);
y=net(xTest);
plotconfusion(tTest,y);
classAcc2=100*(1-confusion(tTest,y))
view(net)

% Compare with normal neural network (3 hidden layers)
net = patternnet([100 81 64]);
net=train(net,xTrain,tTrain);
y=net(xTest);
plotconfusion(tTest,y);
classAcc3=100*(1-confusion(tTest,y))
view(net)


%% Compare with normal neural network (4 hidden layers)
net = patternnet([100 81 64 49]);
net=train(net,xTrain,tTrain);
y=net(xTest);
plotconfusion(tTest,y);
classAcc4=100*(1-confusion(tTest,y))
view(net)

%% plotting abstractions
plot_features(xTrainImages, {feat1, feat2, feat3, feat4}, 10)

%% Extra stuff
hiddenSizeX = 100;
autoencX = trainAutoencoder(xTrainImages,hiddenSizeX, ...
    'MaxEpochs',150, ...  % few, but too slow otherwise
    'L2WeightRegularization',0.0005, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

figure;
plotWeights(autoencX);
featX = encode(autoencX,xTrainImages);

%%
function [] = plot_features(original, feats, i)
% i is the sample
    j = 1;
    subplot(1, length(feats)+1, j)
    imshow(original{i});
    for feat=feats
        j = j + 1;
        subplot(1,length(feats) + 1,j);
        shape_sz = sqrt(length(feat{1}(:,i)));
        minx = min(feat{1}(:,i));
        maxx = max(feat{1}(:,i));
        imshow(reshape(feat{1}(:,i), [shape_sz, shape_sz]), [], 'InitialMagnification', 800);
    end
end