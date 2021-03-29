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
    else % NN is training %TODO there is no method by which w&b can switch signs
        winner = weights.one(:,:,NN.winningIndex);
        weights.one = winner + winner .* NN.mutationCoef .* ...
            (rand(NN.num_inputs,NN.num_neurons_HL1,NN.runsPerGeneration)-.5);
        weights.one(:,:,1) = winner; % keep the first run's weights the same
        winner = biases.one(:,NN.winningIndex);
        biases.one = winner + winner .* NN.mutationCoef .* ...
            (rand(NN.num_neurons_HL1,NN.runsPerGeneration)-.5);
        biases.one(:,1) = winner;
        if NN.enable_HL2
            winner = weights.two(:,:,NN.winningIndex);
            weights.two = winner + winner .* NN.mutationCoef .* ...
                (rand(NN.num_neurons_HL1,NN.num_neurons_HL2,NN.runsPerGeneration)-.5);
            weights.two(:,:,1) = winner;
            winner = biases.two(:,NN.winningIndex);
            biases.two = winner + winner .* NN.mutationCoef .* ...
                (rand(NN.num_neurons_HL2,NN.runsPerGeneration)-.5);
            biases.two(:,1) = winner;
            if ~NN.enable_HL3
                winner = weights.out(:,:,NN.winningIndex);
                weights.out = winner + winner .* NN.mutationCoef .* ...
                    (rand(NN.num_neurons_HL2,NN.num_outputs,NN.runsPerGeneration)-.5);
                weights.out(:,:,1) = winner;
            end
            if NN.enable_HL3
                winner = weights.three(:,:,NN.winningIndex);
                weights.three = winner + winner .* NN.mutationCoef .* ...
                    (rand(NN.num_neurons_HL2,NN.num_neurons_HL3,NN.runsPerGeneration)-.5);
                weights.three(:,:,1) = winner;
                winner = weights.out(:,:,NN.winningIndex);
                weights.out = winner + winner .* NN.mutationCoef .* ...
                    (rand(NN.num_neurons_HL3,NN.num_outputs,NN.runsPerGeneration)-.5);
                weights.out(:,:,1) = winner;
                winner = biases.three(:,NN.winningIndex);
                biases.three = winner + winner .* NN.mutationCoef .* ...
                    (rand(NN.num_neurons_HL3,NN.runsPerGeneration)-.5);
                biases.three(:,1) = winner;
            end
        else
            winner = weights.out(:,:,NN.winningIndex);
            weights.out = winner + winner .* NN.mutationCoef .* ...
                (rand(NN.num_neurons_HL1,NN.num_outputs,NN.runsPerGeneration)-.5);
            weights.out(:,:,1) = winner;
        end
        winner = biases.out(:,NN.winningIndex);
        biases.out = winner + winner .* NN.mutationCoef .* ...
            (rand(NN.num_outputs,NN.runsPerGeneration)-.5);
        biases.out(:,1) = winner;
    end
end
