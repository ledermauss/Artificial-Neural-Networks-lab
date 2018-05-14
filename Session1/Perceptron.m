P = [2 1 -2 -1; 2 -2 2 1];  % data points = cols;
% four data point, two attributesn
T = [0 1 0 1];

% net = perceptron;
% 
% [net, tr_descr] = train(net, P, T);
% 
% net.IW{1,1};
% net.b{1,1};

% Test: I would get the training results if I did
% net.IW{1,1} * P + net.b{1,1}

%% Random initialization
net_random = perceptron;

% configure the net for the train data (dimensions of W and b)
net_random = configure(net_random, P, T); 
% manuall assignment (initial state)
net_random.IW{1,1} = rand(1,2); % 1, 2 is the size of the matrix
net_random.b{1,1} = 0.5;
% train setting the max epochs (around 5 suffice here)
net_random.trainParam.epochs = 20;
[net_random, tr_descr_random] = train(net_random, P, T);
% new bias and W (both change because of training)
bias = net_random.b{1,1}
W = net_random.IW{1,1}
% test on new data
X = [1; -0.3];
X_t = sim(net_random, X);

% again: net.IW{1,1{ * P + net.b{1,1} returns T



