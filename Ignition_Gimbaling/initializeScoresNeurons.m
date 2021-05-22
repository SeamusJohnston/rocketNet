%%
function [score, neurons] = initializeScoresNeurons(NN)
    neurons = struct('one', [] , ...
                     'two', [] , ...
                     'three', [] , ...
                     'out', []);
    neurons.one = zeros(1, NN.num_neurons_HL1,'single'); % maybe add in NN.runsPerGeneration in the future
    neurons.two = zeros(1, NN.num_neurons_HL2, 'single');
    neurons.three = zeros(1, NN.num_neurons_HL3, 'single');
    neurons.out = zeros(1, NN.num_outputs, 'single');
    if NN.isTraining
        score = zeros(NN.runsPerGeneration, 1, 'single');
    else
        score = zeros(NN.numInitialGuesses, 1, 'single');
    end
end