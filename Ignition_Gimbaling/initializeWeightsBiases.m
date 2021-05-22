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
        weights.one =rand(NN.num_inputs,NN.num_neurons_HL1,NN.numInitialGuesses,'single');
        biases.one =rand(NN.num_neurons_HL1,NN.numInitialGuesses,'single');
        if NN.enable_HL2
            weights.two = rand(NN.num_neurons_HL1,NN.num_neurons_HL2,NN.numInitialGuesses,'single');
            biases.two = rand(NN.num_neurons_HL2,NN.numInitialGuesses,'single');
            if ~NN.enable_HL3
                weights.out = rand(NN.num_neurons_HL2,NN.num_outputs,NN.numInitialGuesses,'single');
            end
            if NN.enable_HL3
                weights.three = rand(NN.num_neurons_HL2,NN.num_neurons_HL3,NN.numInitialGuesses,'single');
                weights.out = rand(NN.num_neurons_HL3,NN.num_outputs,NN.numInitialGuesses,'single');
                biases.three = rand(NN.num_neurons_HL3,NN.numInitialGuesses,'single');
            end
        else
            weights.out = rand(NN.num_neurons_HL1,NN.num_outputs,NN.numInitialGuesses,'single');
        end
        biases.out = rand(NN.num_outputs,NN.numInitialGuesses,'single');
    else % NN is training
        winner = weights.one(:,:,NN.winningIndex);
        random = rand(NN.num_inputs, NN.num_neurons_HL1-size(winner,2));
        winner = [winner random];
        weights.one = winner + winner .* NN.mutationCoef .* ...
            (rand(NN.num_inputs,NN.num_neurons_HL1,NN.runsPerGeneration)-.5)*2;
        weights.one(:,:,1) = winner; % keep the first run's weights the same
        
        winner = biases.one(:,NN.winningIndex);
        random = rand(NN.num_neurons_HL1-size(winner,1),1);
        winner = [winner; random];
        biases.one = winner + winner .* NN.mutationCoef .* ...
            (rand(NN.num_neurons_HL1,NN.runsPerGeneration)-.5)*2;
        biases.one(:,1) = winner;
        if NN.enable_HL2
            try
                winner = weights.two(:,:,NN.winningIndex);
            catch
                winner = [];
            end
            random = rand(NN.num_neurons_HL1-size(winner,1), NN.num_neurons_HL2);
            winner = [winner; random];
            random = rand(NN.num_neurons_HL1, NN.num_neurons_HL2-size(winner,2));
            winner = [winner random];
            weights.two = winner + winner .* NN.mutationCoef .* ...
                (rand(NN.num_neurons_HL1,NN.num_neurons_HL2,NN.runsPerGeneration)-.5)*2;
            weights.two(:,:,1) = winner;
            
            try
                winner = biases.two(:,NN.winningIndex);
            catch
                winner = [];
            end
            random = rand(NN.num_neurons_HL2-size(winner,1),1);
            winner = [winner; random];
            biases.two = winner + winner .* NN.mutationCoef .* ...
                (rand(NN.num_neurons_HL2,NN.runsPerGeneration)-.5)*2;
            biases.two(:,1) = winner;
            if ~NN.enable_HL3
                if size(weights.out(:,1,NN.winningIndex),1) ~= NN.num_neurons_HL2
                    winner = [];
                else
                    winner = weights.out(:,:,NN.winningIndex);
                end
                random = rand(NN.num_neurons_HL2, NN.num_outputs-size(winner,2));
                winner = [winner random];
                weights.out = winner + winner .* NN.mutationCoef .* ...
                    (rand(NN.num_neurons_HL2,NN.num_outputs,NN.runsPerGeneration)-.5)*2;
                weights.out(:,:,1) = winner;
            end
            if NN.enable_HL3
                winner = weights.three(:,:,NN.winningIndex);
                random = rand(NN.num_neurons_HL2-size(winner,1), NN.num_neurons_HL3);
                winner = [winner; random];
                random = rand(NN.num_neurons_HL2, NN.num_neurons_HL3-size(winner,2));
                winner = [winner random];
                weights.three = winner + winner .* NN.mutationCoef .* ...
                    (rand(NN.num_neurons_HL2,NN.num_neurons_HL3,NN.runsPerGeneration)-.5)*2;
                weights.three(:,:,1) = winner;
                
                winner = weights.out(:,:,NN.winningIndex);
                random = rand(NN.num_neurons_HL3-size(winner,1),NN.num_outputs);
                winner = [winner; random];
                weights.out = winner + winner .* NN.mutationCoef .* ...
                    (rand(NN.num_neurons_HL3,NN.num_outputs,NN.runsPerGeneration)-.5)*2;
                weights.out(:,:,1) = winner;
                
                winner = biases.three(:,NN.winningIndex);
                random = rand(NN.num_neurons_HL3-size(winner,1),1);
                winner = [winner; random];
                biases.three = winner + winner .* NN.mutationCoef .* ...
                    (rand(NN.num_neurons_HL3,NN.runsPerGeneration)-.5)*2;
                biases.three(:,1) = winner;
            end
        else
            winner = weights.out(:,:,NN.winningIndex);
            random = rand(NN.num_neurons_HL1-size(winner,1),NN.num_outputs);
            winner = [winner; random];
            weights.out = winner + winner .* NN.mutationCoef .* ...
                (rand(NN.num_neurons_HL1,NN.num_outputs,NN.runsPerGeneration)-.5)*2;
            weights.out(:,:,1) = winner;
        end
        winner = biases.out(:,NN.winningIndex);
        biases.out = winner + winner .* NN.mutationCoef .* ...
            (rand(NN.num_outputs,NN.runsPerGeneration)-.5)*2;
        biases.out(:,1) = winner;
    end
end
