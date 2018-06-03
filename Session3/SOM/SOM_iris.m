clear
clc
close all

%% Load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load iris
X = iris(:,1:end-1);
true_labels = iris(:,end); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Training the SOM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iter = 300;
initHood = 3;
topologyFcn = 'hextop';    %  'hextop'(*), 'gridtop' and 'randtop'
distanceFcn = 'linkdist';  % 'linkdist'(*), 'dist' and 'boxdist'

x_length = 3;
y_length = 1;
gridsize=[y_length x_length];
net = selforgmap(gridsize, iter, initHood, topologyFcn, distanceFcn);
net = train(net,X');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Assigning examples to clusters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
outputs = sim(net,X');
[~,assignment]  =  max(outputs);

plot_options = []
plot_options.labels =  fliplr(true_labels')'
%plot_options.labels =  assignment

ml_plot_data(X, plot_options)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Compare clusters with true labels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ARI=RandIndex(assignment,true_labels);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%
%% Grid searching %%
%%%%%%%%%%%%%%%%%%%%%
initHood = 100;
max_exp = 20;
topologies = {'hextop', 'gridtop', 'randtop'};
distances = {'linkdist', 'dist', 'boxdist', 'mandist'};

ARI_matrix = zeros(length(topologies),length(distances));

t = 0;
for topology = topologies
    t = t + 1
    d = 0
    for distance = distances
        d = d + 1
        ARI_vec = [];
        for e=1:max_exp
            net = selforgmap(gridsize, iter, initHood, topology{1}, distance{1});
            net.trainParam.showWindow = false;
            net = train(net,X');
            outputs = sim(net,X');
            [~,assignment]  =  max(outputs);
            ARI = RandIndex(assignment, true_labels);
            ARI_vec = [ARI_vec; ARI];
        end
        ARI = mean(ARI_vec);
        ARI_matrix(t, d) = ARI
    end
end

%% Different epochs

topologyFcn = 'hextop';    %  'hextop'(*), 'gridtop' and 'randtop'
distanceFcn = 'dist';  % 'linkdist'(*), 'dist', 'mandist' 'boxdist'
net = selforgmap([5 5], iter, initHood, topologyFcn, distanceFcn)
% plot before training
net = configure(net, X')
plotsompos(net, X'), set(gca, 'FontSize', 14)

for epochs = [1, 5, 100]
net.trainParam.epochs = epochs;
raul = train(net, X')
outputs = sim(raul,X');
[~,assignment]  =  max(outputs);
%plotsompos(raul, X'), set(gca, 'FontSize', 14)
ARI=RandIndex(assignment,true_labels)
pause()
end

%% Different neurons
topologyFcn = 'hextop';    %  'hextop'(*), 'gridtop' and 'randtop'
distanceFcn = 'mandist';  % 'linkdist'(*), 'dist' and 'boxdist'
net = selforgmap([3 1], iter, initHood, topologyFcn, distanceFcn)
% plot before training
net = train(net, X')
plotsompos(net, X'), set(gca, 'FontSize', 14)