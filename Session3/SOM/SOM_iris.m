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
plot_options.labels =  true_labels
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
distances = {'linkdist', 'dist', 'boxdist'};
iters = [100, 150, 200, 250, 300, 350, 400, 450, 500];

ARI_matrix = zeros(length(iters), length(distances), length(topologies));
t = 0;
d = 0;
i = 0;

for topology = topologies
    t = t + 1
    i = 0;
    for iter = iters
        i = i + 1
        d = 0;
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
            ARI_matrix(i, d, t) = ARI
        end
    end
end

%%
iter = 100;
initHood = 3;
topologyFcn = 'hextop';    %  'hextop'(*), 'gridtop' and 'randtop'
distanceFcn = 'linkdist';  % 'linkdist'(*), 'dist' and 'boxdist'
net = selforgmap([3 3], iter, initHood, topologyFcn, distanceFcn)
% plot before training
net = configure(net, X')
plotsompos(net,X')
