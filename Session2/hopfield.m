T = [1 1; -1 -1; 1 -1; 2 2; -2 -2; -3 3; 4 4; 9 4]'
net = newhop(T);
%%
% initial vectors. Size N
net = newhop(T);

Ai = {rand(2,1)};
Ai = {[0.5 0]' }; % case of spurious state: Y converges to [1 0], not an attractor
% it makes sense, it is a remaining one on the square

Y = net([],[],Ai);
Y = net({10}, {}, Ai);
Y{10}

%%
T = [1 1 ; -1 -1; -1 1]'
net = newhop(T);

% init
% Ai = {rand(2,1) .* 2 - 1};
Ai = {[9; 4.5]};
%Y = net([],[],Ai);
Y = net({50}, {}, Ai)

Ynew = Ai{1};

%%
step = 0;
convergences = 0;
%while Ynew ~= Y
while convergences < 10
    Y = Ynew;
    Ynew = net([], [], Y);
    if Ynew==Y
        convergences = convergences + 1;
    end
    step = step+1;
end
step
Y

%%
Ai = {[-0.9; -0.8; 0.7]};
[Y,Pf,Af] = net({1 5},{},Ai);
     Y{1}
