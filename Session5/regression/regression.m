load  Data_Problem1_regression.mat

Tnew=(9.*T1 + 6.*T2 + 6.* T3+ 5.*T4 + 1.*T5)/(9 + 6 + 6 + 5 + 1)

%%
F = scatteredInterpolant(X1,X2,Tnew);

[xq,yq] = meshgrid(0:0.01:1);
vq1 = F(xq,yq);
figure;
mesh(xq,yq,vq1), xlabel("X1"), ylabel("X2"), zlabel("Tnew")

%% defining sets
X = [X1 X2];
idx = randperm(length(Tnew));
Xtrain = X(idx(1:1000),:);
Ytrain = Tnew(idx(1:1000),:);
Xtest = X(idx(1001:2000),:);
Ytest = Tnew(idx(1001:2000),:);
Xval = X(idx(2001:3000),:);
Yval = Tnew(idx(2001:3000),:);
X = X(idx(1:3000),:);
Y = Tnew(idx(1:3000),:);

%%
learnAlg = 'trainlm'
nHidden = 50;
maxEpochs = 500;
NN1 = NN1Hidden(nHidden, 'trainlm', 500, X', Y', 'tansig', false);
%%
fcnTransfer = {'logsig'}
experiments = 5;
neurons = [25 50 100 150 200 250 300]
i = 1;
res = zeros(length(neurons), 6, length(fcnTransfer));
for transfer= fcnTransfer
    transResults = [];
    for nhidden = neurons
        neuronExps = zeros(experiments, 5);
        for e=1:experiments
            % 1000 iterations, let it converge
            NN = NN1Hidden(nHidden, 'trainlm', 200, X', Y', transfer{1}, false);
            [~, ~, R] = NN.testRegression();
            neuronExps(e,:) = [R NN.testMSE NN.trainMSE NN.epochs NN.time];
        end
        avgExps = mean(neuronExps);
        transResults = [transResults; nhidden  avgExps] % length(neurons) * 6 array
    end
    res(:,:,i) = transResults;
    i = i + 1;
end
%%
NN = NN1Hidden(25, 'trainlm', 500, X', Y', 'logsig', true);
%% Plot sim interpolatted
Y_sim = sim(NN.net, Xtest');
F_sim = scatteredInterpolant(Xtest(:,1),Xtest(:,2),Y_sim');

[xq_sim,yq_sim] = meshgrid(0:0.01:1);
vq_sim = F(xq_sim,yq_sim);
figure;
hold on
hs1 = surf(xq_sim,yq_sim,vq_sim), xlabel("X1"), ylabel("X2"), zlabel("Tnew"),
set(hs1,'FaceColor',[1 0 0],'FaceAlpha',0.5);
surf(xq,yq,vq1),
set(gca,'FaceColor',[0 1 0],'FaceAlpha',0.5);
hold off


%% scatter
figure;
scatter3(Xtest(:,1), Xtest(:,2), Y_sim', 'go'), xlabel("X1"), ylabel("X2"), zlabel("Tnew")
figure;
scatter3(Xtest(:,1), Xtest(:,2), Ytest, 'ro'), xlabel("X1"), ylabel("X2"), zlabel("Tnew")

