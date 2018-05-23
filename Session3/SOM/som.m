x = simplecluster_dataset;
%x = x + rand(2,1000)*2
plot(x(1,:)', x(2,:)', 'o')

%%
iter = 100;
initHood = 3;
topologyFcn = 'hextop';    %  'hextop'(*), 'gridtop' and 'randtop'
distanceFcn = 'linkdist';  % 'linkdist'(*), 'dist' and 'boxdist'
% [2 2] is very interesting: it clusters the data
net = selforgmap([8 8], iter, initHood, topologyFcn, distanceFcn);
net = train(net,x);
view(net)
y = net(x);
classes = vec2ind(y);

%%
uClass = unique(classes)

pntColor = hsv(length(uClass))
figure,hold on
for ind = 1:length(uClass)
    scatter(x(1,classes == uClass(ind)), x(2,classes == uClass(ind)), 150, 'MarkerFaceColor',pntColor(ind,:),'Marker','*')
end

     