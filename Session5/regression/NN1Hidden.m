classdef NN1Hidden

    
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
        epochs
        time
    end
    
    methods
        function obj = NN1Hidden(nHidden, learnAlg, maxEpochs, x, y, transfer, window)
            % produces already trained network
            % here: W (V * x(NHidden*1)) (1*1)
            net = feedforwardnet(nHidden, learnAlg);
            net.trainParam.epochs =  maxEpochs; % to make it shorter
            net.trainParam.showWindow = window; % avoid showing training window
                % go until last iteration, let it overfit
            net.divideParam.valRatio = 0.3333;
            net.divideParam.testRatio = 0.3333;
            net.divideParam.trainRatio = 0.3333;
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
            obj.epochs = res.best_epoch;
            obj.time = res.time(end); % time on last epoch, not best
        end
  
                
        function simData = simulateData(obj)
            simData = sim(obj.net, obj.x);
        end
        
        function [testSet, simTest] = simulateTest(obj)
            testSet = obj.xtest;
            simTest = sim(obj.net, testSet);
        end
        
        function [trainSet, simTrain] = simulateTrain(obj)
            trainSet = obj.xtrain;
            simTrain = sim(obj.net, trainSet);
        end
        
        function [m,b,R] = testRegression(obj)
            testSim = sim(obj.net, obj.xtest);
            testTarget = obj.ytest;
            [m, b, R] = postreg(testSim, testTarget);
        end
        
         function [m,b,R] = trainRegression(obj)
            trainSim = sim(obj.net, obj.xtrain);
            trainTarget = obj.ytrain;
            [m, b, R] = postreg(trainSim, trainTarget);
        end
    end
end

