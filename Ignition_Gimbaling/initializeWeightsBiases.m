function [weights, biases] = initializeWeightsBiases(NN)
    weights = struct('one', [] , ...
                     'two', [] , ...
                     'three', [] , ...
                     'out', []);
    biases  = struct('one', [] , ...
                     'two', [] , ...
                     'three', [] , ...
                     'out', []);
    weights.one = -.5 + rand(NN.num_inputs,NN.num_neurons_HL1,NN.runsPerGeneration,'single');
    biases.one = -.5 + rand(NN.num_neurons_HL1,NN.runsPerGeneration,'single');
    if NN.enable_HL2
        weights.two = -.5 + rand(NN.num_neurons_HL1,NN.num_neurons_HL2,NN.runsPerGeneration,'single');
        biases.two = -.5 + rand(NN.num_neurons_HL2,NN.runsPerGeneration,'single');
        if ~NN.enable_HL3
            weights.out = -.5 + rand(NN.num_neurons_HL2,NN.num_outputs,NN.runsPerGeneration,'single');
        end
        if NN.enable_HL3
            weights.three = -.5 + rand(NN.num_neurons_HL2,NN.num_neurons_HL3,NN.runsPerGeneration,'single');
            weights.out = -.5 + rand(NN.num_neurons_HL3,NN.num_outputs,NN.runsPerGeneration,'single');
            biases.three = -.5 + rand(NN.num_neurons_HL3,NN.runsPerGeneration,'single');
        end
    else
        weights.out = -.5 + rand(NN.num_neurons_HL1,NN.num_outputs,NN.runsPerGeneration,'single');
    end
    biases.out = -.5 + rand(NN.num_outputs,NN.runsPerGeneration,'single');

%     
% 
%     if isMutating == true
%         weight = weight(:,:,:,maxIndex) + .01*(-.5 + rand(hiddenLayers+1,neuronsPerLayer,...
% neuronsPerLayer,initialGuesses,'single'));
%         bias = bias(:,:,:,maxIndex) + .01*(-.5 + rand(hiddenLayers+1,neuronsPerLayer,...
% neuronsPerLayer,initialGuesses,'single'));
%     end
end
