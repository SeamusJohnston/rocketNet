classdef rotateNet
    %ROTATENET Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        lr
        epsilon % exploration rate
        eps_decay
        layers
    end
    
    methods
        function obj = rotateNet(learningRate, numHiddenLayers, hiddenLayerSize, epsilon, eps_decay)
            %ROTATENET Construct an instance of this class
            %   Detailed explanation goes here
            inSize = 6;
            outSize = 2;
            obj.lr = learningRate;
            obj.epsilon = epsilon;
            obj.eps_decay = eps_decay;
            obj.layers{1} = netLayer(inSize, hiddenLayerSize, 'relu', obj.lr);
            for i = 1:numHiddenLayers
                obj.layers{1+i} = netLayer(hiddenLayerSize, hiddenLayerSize, 'relu', obj.lr);
            end
            obj.layers{2+numHiddenLayers} = netLayer(hiddenLayerSize, outSize, 'linear', obj.lr);
        end
        
        function u = determineAction(obj, x)
            u = [0;0];
            if (rand() > obj.epsilon)
                u = obj.forward(x); % argmax here in article?
            else
                u = rand(2,1)*pi - pi/2; % rand action between -pi/2 and pi/2
            end
        end
        
        function u = forward(obj, x)
            out = x;
            for i = 1:length(obj.layers)
                out = obj.layers{i}.forward(out);
            end
            u = out;
        end
        
        function obj = decay(obj)
            if (obj.epsilon > 0.01)
                obj.epsilon = obj.epsilon * (1-obj.eps_decay);
            end
        end
    end
end

