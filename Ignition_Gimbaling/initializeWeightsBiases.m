function [weights, biases] = initializeWeightsBiases(NN, weights, biases)
    if ~NN.isTraining
        weights = struct('one', [] , ...
                         'two', [] , ...
                         'three', [] , ...
                         'out', []);
        biases  = struct('one', [] , ...
                         'two', [] , ...
                         'three', [] , ...
                         'out', []);
        weights.one = -.5 + rand(NN.num_inputs,NN.num_neurons_HL1,NN.numInitialGuesses,'single');
        biases.one = -.5 + rand(NN.num_neurons_HL1,NN.numInitialGuesses,'single');
        if NN.enable_HL2
            weights.two = -.5 + rand(NN.num_neurons_HL1,NN.num_neurons_HL2,NN.numInitialGuesses,'single');
            biases.two = -.5 + rand(NN.num_neurons_HL2,NN.numInitialGuesses,'single');
            if ~NN.enable_HL3
                weights.out = -.5 + rand(NN.num_neurons_HL2,NN.num_outputs,NN.numInitialGuesses,'single');
            end
            if NN.enable_HL3
                weights.three = -.5 + rand(NN.num_neurons_HL2,NN.num_neurons_HL3,NN.numInitialGuesses,'single');
                weights.out = -.5 + rand(NN.num_neurons_HL3,NN.num_outputs,NN.numInitialGuesses,'single');
                biases.three = -.5 + rand(NN.num_neurons_HL3,NN.numInitialGuesses,'single');
            end
        else
            weights.out = -.5 + rand(NN.num_neurons_HL1,NN.num_outputs,NN.numInitialGuesses,'single');
        end
        biases.out = -.5 + rand(NN.num_outputs,NN.numInitialGuesses,'single');
    else % NN is training    
        weights.one = weights.one(:,:,NN.winningIndex) + NN.mutationCoef*(-.5 + rand(NN.num_inputs,NN.num_neurons_HL1,NN.runsPerGeneration,'single'));
        biases.one = biases.one(:,NN.winningIndex) + NN.mutationCoef*(-.5 + rand(NN.num_neurons_HL1,NN.runsPerGeneration,'single'));
        if NN.enable_HL2
            weights.two = weights.two(:,:,NN.winningIndex) + NN.mutationCoef*(-.5 + rand(NN.num_neurons_HL1,NN.num_neurons_HL2,NN.runsPerGeneration,'single'));
            biases.two = biases.two(:,NN.winningIndex) + NN.mutationCoef*(-.5 + rand(NN.num_neurons_HL2,NN.runsPerGeneration,'single'));
            if ~NN.enable_HL3
                weights.out = weights.out(:,:,NN.winningIndex) + NN.mutationCoef*(-.5 + rand(NN.num_neurons_HL2,NN.num_outputs,NN.runsPerGeneration,'single'));
            end
            if NN.enable_HL3
                weights.three = weights.three(:,:,NN.winningIndex) + NN.mutationCoef*(-.5 + rand(NN.num_neurons_HL2,NN.num_neurons_HL3,NN.runsPerGeneration,'single'));
                weights.out = weights.out(:,:,NN.winningIndex) + NN.mutationCoef*(-.5 + rand(NN.num_neurons_HL3,NN.num_outputs,NN.runsPerGeneration,'single'));
                biases.three = biases.three(:,NN.winningIndex) + NN.mutationCoef*(-.5 + rand(NN.num_neurons_HL3,NN.runsPerGeneration,'single'));
            end
        else
            weights.out = weights.out(:,:,NN.winningIndex) + NN.mutationCoef*(-.5 + rand(NN.num_neurons_HL1,NN.num_outputs,NN.runsPerGeneration,'single'));
        end
        biases.out = biases.out(:,NN.winningIndex) + NN.mutationCoef*(-.5 + rand(NN.num_outputs,NN.runsPerGeneration,'single'));
    end
end
