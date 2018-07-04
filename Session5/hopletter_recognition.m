%A script to test if 2d Hopfield network can recognize letter of alphabet



%function [] = hopletter_recognition()

 close all
%%
data = prprob;
let1 = data(:, 1);
let1 = imresize(reshape(let1, 5, 7), 1, 'nearest'); let1 = let1(:);
let2 = data(:, 2);
let2 = imresize(reshape(let2, 5, 7), 1, 'nearest'); let2 = let2(:);
let3 = data(:, 3);
let3 = imresize(reshape(let3, 5, 7), 1, 'nearest'); let3 = let3(:);
let4 = data(:, 4);
let4 = imresize(reshape(let4, 5, 7), 1, 'nearest'); let4 = let4(:);
let5 = data(:, 5);
let5 = imresize(reshape(let5, 5, 7), 1, 'nearest'); let5 = let5(:);
let6 = data(:, 6);
let6 = imresize(reshape(let6, 5, 7), 1, 'nearest'); let6 = let6(:);
let7 = data(:, 7);
let7 = imresize(reshape(let7, 5, 7), 1, 'nearest'); let7 = let7(:);
let8 = data(:, 8);
let8 = imresize(reshape(let8, 5, 7), 1, 'nearest'); let8 = let8(:);
let9 = data(:, 9);
let9 = imresize(reshape(let9, 5, 7), 1, 'nearest'); let9 = let9(:);
let10 = data(:, 10);
let10 = imresize(reshape(let10, 5, 7), 1, 'nearest'); let10 = let10(:);
let11 = data(:, 11);
let11 = imresize(reshape(let11, 5, 7), 1, 'nearest'); let11 = let11(:);
let12 = data(:, 12);
let12 = imresize(reshape(let12, 5, 7), 1, 'nearest'); let12 = let12(:);
let13 = data(:, 13);
let13 = imresize(reshape(let13, 5, 7), 1, 'nearest'); let13 = let13(:);
let14 = data(:, 14);
let14 = imresize(reshape(let14, 5, 7), 1, 'nearest'); let14 = let14(:);
let15 = data(:, 15);
let15 = imresize(reshape(let15, 5, 7), 1, 'nearest'); let15 = let15(:);
let16 = data(:, 16);
let16 = imresize(reshape(let16, 5, 7), 1, 'nearest'); let16 = let16(:);
let17 = data(:, 17);
let17 = imresize(reshape(let17, 5, 7), 1, 'nearest'); let17 = let17(:);
let18 = data(:, 18);
let18 = imresize(reshape(let18, 5, 7), 1, 'nearest'); let18 = let18(:);
let19 = data(:, 19);
let19 = imresize(reshape(let19, 5, 7), 1, 'nearest'); let19 = let19(:);
let20= data(:, 20);
let20 = imresize(reshape(let20, 5, 7), 1, 'nearest'); let20 = let20(:);
let21= data(:, 21);
let21= imresize(reshape(let21, 5, 7), 1, 'nearest'); let21 = let21(:);
let22= data(:, 22);
let22= imresize(reshape(let22, 5, 7), 1, 'nearest'); let22 = let22(:);
let23= data(:, 23);
let23= imresize(reshape(let23, 5, 7), 1, 'nearest'); let23 = let23(:);
let24= data(:, 24);
let24= imresize(reshape(let24, 5, 7), 1, 'nearest'); let24 = let24(:);
let25= data(:, 25);
let25= imresize(reshape(let25, 5, 7), 1, 'nearest'); let25 = let25(:);

X = [let1, let2, let3, let4, let5, let6, let7, let8, let9, let10, let11, let12, let13, let14, let15, let16, let17, let18, let19, let20, let21, let22 let23, let24, let25];

%%
figure;
for i=1:10
    subplot(2, 5, i);
    imshow(reshape(X(:,i), 5, 7)', [0,1], 'InitialMagnification', 800);
end
%% Values must be +1 or -1
X(X==0)=-1;
%% -------------------------------------------------------------------------

%Attractors of the Hopfield network
% w = reshape(X(:,1), 5, 7)';
% imshow(w)

m = X(:,1)';
a = X(:,2)';
u = X(:,3)';
r = X(:,4)';
o = X(:,5)';
p = X(:,6)';
d = X(:,7)';
e = X(:,8)';
l = X(:,9)';
i = X(:,10)';
A = X(:,11)';
B = X(:,12)';
C = X(:,13)';
D = X(:,14)';
E = X(:,15)';
F = X(:,16)';
G = X(:,17)';
H = X(:,18)';
I = X(:,19)';
J = X(:,20)';
K = X(:,21)';
L = X(:,22)';
M = X(:,23)';
N = X(:,24)';
O = X(:,25)';
U = [m;a;u;r;o;p;d;e;l;i;A;B;C;D;E;F;G;H;I;J;K;L;M;N;O]';
%%
error_list = zeros(1,25);

 for num = 1:25
    num_letters = num;

    
    T = U(:,1:num_letters);

    %Create network
    net = newhop(T);


    %Check if digits are attractors
    [Y,~,~] = sim(net,num_letters,[],T);
    Y = Y';

    figure;

    %subplot(num_letters,3,1);


    for i = 1:num_letters
        letter = Y(i,:);
        letter = reshape(letter,5,7)'; 

        %subplot(num_letters,3,((i-1)*3)+1);
        %imshow(letter)
        if i == 1
            title('Attractors')
        end
        hold on
    end
    %%
    % Add noise - randmonly flip 3 pixels
    % as pixels are chosen randomly and the task does not mention
    % that they have to be different, they might be the same
    % in that case pixel is inverted twice i.e. stays the same.
    % in the unlikely event of choosing the same pixel 3 times, only this
    % pixel will be inverted

    %% choose pixels to flip
    [dim, N] = size(X);
    R = round(unifrnd(0,35,3,N));
    %flip pixels
    Xn = X;
    for i=1:N;
        for j=1:dim
            if any(j==R(:,i))
                Xn(j,i) = -X(j,i);
            else
                Xn(j,i) = X(j,i);
            end
        end
    end

    %% Show noisy digits:

    subplot(num_letters,3,2);

    for i = 1:num_letters
    letter = Xn(:,i);
    letter = reshape(letter,5,7)';
    %subplot(num_letters,3,((i-1)*3)+2);
    %imshow(letter)
    if i == 1
        title('Noisy digits')
    end
    hold on
    end

%% ------------------------------------------------------------------------

    %See if the network can correct the corrupted digits 


    num_steps = 20;

    Tn = Xn(:,1:num_letters);
    [Yn,~,~] = sim(net,{num_letters num_steps},{},Tn);
    Yn = Yn{1,num_steps};
    Yn = Yn';
    %subplot(num_letters,3,2);
    [dim1, dim2] = size(Yn);
    for i = 1:dim2
        for j =1:dim1
            if Yn(j,i) > 0
                Yn(j,i) = 1;
            else
                Yn(j,i) = -1;
            end
        end
    end
%%
    for i = 1:num_letters
        letter = Yn(i,:);
        letter = reshape(letter,5,7)'; 

        %subplot(num_letters,3,((i-1)*3)+3);
        %imshow(letter,'InitialMagnification', 800)
    if i == 1
        title('Reconstructed noisy digits')
    end
    hold on

    end
%%    
    error = sum(sum(abs((Yn-Y)/2)));
    error_list(num) = error;
end
%%
error_list
figure;
plot(1:length(error_list), error_list), set(gca, 'FontSize', 14), ylabel("Wrong pixels"),
xlabel("Stored letters (P)"), title("Retrieval error", 'Fontsize', 16);
