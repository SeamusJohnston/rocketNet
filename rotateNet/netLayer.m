classdef netLayer
    %NETLAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        act_func % activation function handle
        weights
        lr
        lr_decay
        backward_store_in
        backward_store_out
    end
    
    methods
        function obj = netLayer(in_size, out_size, activation, lr, lr_decay)
            %NETLAYER Construct an instance of this class
            %   Detailed explanation goes here
            obj.weights = rand(out_size, in_size+1) - 0.5;
            obj.act_func = activation;
            obj.lr = lr;
            obj.lr_decay = lr_decay;
        end
        
        function [obj,out] = forward(obj,u, remember)
            % Add bias term
            u = [u; 1;];
            unact_out = obj.weights * u;
            if strcmp(obj.act_func, 'relu')
                out = obj.relu(unact_out);
            else 
                out = obj.linear(unact_out);
            end
            
            if remember
                obj.backward_store_in = u;
                obj.backward_store_out = unact_out;
            end
        end
        
        function [obj, out] = backward(obj, gradientFromAbove)
            adjust_mul = gradientFromAbove;
            if strcmp(obj.act_func, 'relu')
                adjust_mul = reluDerivative(obj.backward_store_out) * gradientFromAbove;
            end
            D_i = dot(obj.backward_store_in', adjust_mul');
            delta_i = dot(adjust_mul, obj.weights');
            obj = obj.updateWeights(D_i);
            out = delta_i;
        end
                
        
        function y = linear(~, x)
            y = x;
        end
        
        function y = relu(~, x)
            y = x .* (x>0);
        end
        
        function y = reluDerivative(~, x)
            y = 1 .* (x>0);
        end
        
        function obj = updateWeights(obj, gradient)
            obj.weights = obj.weights - gradient * obj.lr;
        end
        
        function obj = decay(obj)
            if (obj.lr > 0.01)
                obj.lr = obj.lr * (1-obj.lr_decay);
            end
        end
    end
end

