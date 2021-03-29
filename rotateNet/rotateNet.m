classdef rotateNet
    %ROTATENET Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        lr
        epsilon % exploration rate
        eps_decay
        layers
        memory
        memories
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
            obj.layers{1} = netLayer(inSize, hiddenLayerSize, 'relu', obj.lr, eps_decay);
            for i = 1:numHiddenLayers
                obj.layers{1+i} = netLayer(hiddenLayerSize, hiddenLayerSize, 'relu', obj.lr, eps_decay);
            end
            obj.layers{2+numHiddenLayers} = netLayer(hiddenLayerSize, outSize, 'linear', obj.lr, eps_decay);
            obj.memory = cell(1000000,1);
            obj.memories = 0;
        end
        
        function [obj, u] = determineAction(obj, x)
            u = [0;0];
            if (rand() > obj.epsilon)
                [obj, u] = obj.forward(x, true); % argmax here in article?
            else
                u = rand(2,1)*pi - pi/2; % rand action between -pi/2 and pi/2
            end
        end
        
        function [obj, u] = forward(obj, x, remember)
            out = x;
            for i = 1:length(obj.layers)
                [obj.layers{i},out] = obj.layers{i}.forward(out, remember);
            end
            u = out;
        end
        
        function obj = backward(obj, calc_values, exp_values)
            delta = calc_values - exp_values;
            for i = length(obj.layers):-1:1
                [obj.layers{i}, delta] = obj.layers{i}.backward(delta);
            end
        end
        
        function obj = decay(obj)
            if (obj.epsilon > 0.01)
                obj.epsilon = obj.epsilon * (1-obj.eps_decay);
            end
        end
        
        function obj = remember(obj, done, action, obs, prev_obs, score)
            temp_memory = struct('done', done, 'action', action, 'observation',...
                obs, 'prev_observation', prev_obs, 'score', score);
            obj.memory(1:end-1) = obj.memory(2:end);
            obj.memory{end} = temp_memory;
            obj.memories = min(obj.memories+1, length(obj.memory));
        end
        
        function obj = experience_replay(obj, update_size, rocket, dt)
            if obj.memories >= update_size
                indexes = randi(obj.memories, 1, update_size);
                for i=1:update_size
                    memory = obj.memory{length(obj.memory)-indexes(i)};
                    action_score = memory.score;
                    [~,alternate_action] = obj.forward(memory.prev_observation, false);
                    rocket_copy = rocket;
                    rocket_copy.x = memory.prev_observation;
                    rocket_copy = rocket_copy.stepDynamics(alternate_action, dt);
                    alternate_score = obj.evaluateScore(rocket_copy.x);
                    if alternate_score > action_score
                        obj = obj.backward(memory.action, alternate_action);
                        obj = obj.decay();
                        for j = 1:length(obj.layers)
                            obj.layers{j} = obj.layers{j}.decay();
                        end
                    end
                end
            end
        end
        
        % best score is 0
        function score = evaluateScore(obj, x)
            % Score is based off of landing at 0 degrees
            score = -(abs(x(5)) + 0.25*abs(x(6)))^2;
        end
        
    end
end

