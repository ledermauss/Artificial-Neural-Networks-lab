classdef NN1Pattern

    
    properties
        net
        testMSE
        trainMSE
        x
        y
        xtest
        ytest
        xtrain
        ytrain
        xval
        yval
        epochs
        time
    end
    
    methods
        function obj = NN1Pattern(nHidden, learnAlg, maxEpochs, x, y, transfer, window)
            % produces already trained network
            % here: W (V * x(NHidden*1)) (1*1)
            net = patternnet(nHidden, learnAlg);
            net.trainParam.epochs =  maxEpochs; % to make it shorter
            net.trainParam.showWindow = window; % avoid showing training window
                % go until last iteration, let it overfit
            %net.divideParam.valRatio = 0.3333;
            %net.divideParam.testRatio = 0.3333;
            %net.divideParam.trainRatio = 0.3333;
            net.layers{1}.transferFcn = transfer;
            [obj.net, res] = train(net, x, y);
             % if necessary, set weights already

            obj.testMSE = res.best_tperf; % last epoch
            obj.trainMSE = res.best_perf;
            obj.x = x;
            obj.y = y;
            obj.xtest = x(:,res.testInd);
            obj.ytest = y(:,res.testInd);
            obj.xtrain = x(:,res.trainInd);
            obj.ytrain = y(:,res.trainInd);
            obj.xval = x(:, res.valInd);
            obj.yval = y(:, res.valInd);
            obj.epochs = res.best_epoch;
            obj.time = res.time(end); % time on last epoch, not best
        end
  
                
        function CCR = valCCR(obj)
            ysim = sim(obj.net, obj.xval);
            ysim = ysim > 0.5; % NOTE: valid only with binary classif!!
            CCR = sum(ysim(1,:) == obj.yval(1,:))/length(ysim(1,:));
        end
        function CCR = testCCR(obj)
            ysim = sim(obj.net, obj.xtest);
            ysim = ysim > 0.5; % NOTE: valid only with binary classif!!
            CCR = sum(ysim(1,:) == obj.ytest(1,:))/length(ysim(1,:));
            
        end
        function CCR = trainCCR(obj)
            ysim = sim(obj.net, obj.xtrain);
            ysim = ysim > 0.5; % NOTE: valid only with binary classif!!
            CCR = sum(ysim(1,:) == obj.ytrain(1,:))/length(ysim(1,:));
        end
        
        function [] = testConfusion(obj)
            ysim = sim(obj.net, obj.xtest);
            plotconfusion(obj.ytest, ysim);
        end 
        function [] = trainConfusion(obj)
            ysim = sim(obj.net, obj.xtrain);
            plotconfusion(obj.ytrain, ysim);
        end
        function [] = valConfusion(obj)
            ysim = sim(obj.net, obj.xval);
            plotconfusion(obj.yval, ysim);
        end
    end
end

