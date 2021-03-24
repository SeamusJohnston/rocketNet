classdef netLayer
    %NETLAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        act_func % activation function handle
        weights
        lr
    end
    
    methods
        function obj = netLayer(in_size, out_size, activation, lr)
            %NETLAYER Construct an instance of this class
            %   Detailed explanation goes here
            obj.weights = rand(out_size, in_size+1) - 0.5;
            obj.act_func = activation;
            obj.lr = lr;
        end
        
        function out = forward(obj,u)
            % Add bias term
            u = [u; 1;];
            unact_out = obj.weights * u;
            if strcmp(obj.act_func, 'relu')
                out = obj.relu(unact_out);
            else 
                out = obj.linear(unact_out);
            end
        end
        
        function y = linear(~, x)
            y = x;
        end
        
        function y = relu(~, x)
            y = x .* (x>0);
        end
        
    end
end

