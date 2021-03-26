%%
function [score, neurons] = initializeScoreNeurons(NN)
    neurons = struct('one', [] , ...
                     'two', [] , ...
                     'three', [] , ...
                     'out', []);
    neurons.one = zeros(NN.num_neurons_HL1, 1,'single'); % maybe add in NN.runsPerGeneration in the future
    neurons.two = zeros(NN.num_neurons_HL2, 1, 'single');
    neurons.three = zeros(NN.num_neurons_HL3, 1, 'single');
    neurons.out = zeros(NN.num_outputs, 1, 'single');
    score = zeros(NN.runsPerGeneration, 1, 'single');
end