function [ignite, phi_dot_dot] = runNeuralNet(weights, biases, neurons, state, geometry, NN, sim)
	inputs = [state.phi_dot, ...
             state.theta_dot, ...
             state.vel_x, ...
             state.vel_z, ...
             state.theta, ...
             state.phi, ...
             state.pos_cg(1)-geometry.pos_pad(1), ...
             state.pos_cg(3)-geometry.pos_pad(3)];
% calculate hidden layer 1 neurons
for n = 1:NN.num_neurons_HL1    % each neuron in hidden layer
    for i = 1:size(inputs,2) % number of inputs
        neurons.one(n) = neurons.one(n) + inputs(i)*weights.one(i, n, sim.run);
    end
    neurons.one(n) = neurons.one(n) + biases.one(n, sim.run);
end
% calculate hidden layer 2 neurons
if NN.enable_HL2
    for n = 1:NN.num_neurons_HL2    % each neuron in hidden layer
        for i = 1:NN.num_neurons_HL1 % number of inputs
            neurons.two(n) = neurons.two(n) + neurons.one(i)*weights.two(i, n, sim.run);
        end
        neurons.two(n) = neurons.two(n) + biases.two(n, sim.run);
    end
end
% calculate hidden layer 3 neurons
if NN.enable_HL3
    for n = 1:NN.num_neurons_HL3    % each neuron in hidden layer
        for i = 1:NN.num_neurons_HL2 % number of inputs
            neurons.three(n) = neurons.three(n) + neurons.two(i)*weights.three(i, n, sim.run);
        end
        neurons.three(n) = neurons.three(n) + biases.three(n, sim.run);
    end
end
% calculate output layer neurons
for n = 1:NN.num_outputs    % each neuron in output layer
    if NN.enable_HL3
        for i = 1:NN.num_neurons_HL3 % number of inputs
            neurons.out(n) = neurons.out(n) + neurons.three(i)*weights.out(i, n, sim.run);
        end
        neurons.out(n) = neurons.out(n) + biases.out(n, sim.run);
    elseif NN.enable_HL2
        for i = 1:NN.num_neurons_HL2 % number of inputs
            neurons.out(n) = neurons.out(n) + neurons.two(i)*weights.out(i, n, sim.run);
        end
        neurons.out(n) = neurons.out(n) + biases.out(n, sim.run);
    else
        for i = 1:NN.num_neurons_HL1 % number of inputs
            neurons.out(n) = neurons.out(n) + neurons.one(i)*weights.out(i, n, sim.run);
        end
        neurons.out(n) = neurons.out(n) + biases.out(n, sim.run);
    end
end
ignite      = neurons.out(1); % output(1); % m
phi_dot_dot = neurons.out(2); % output(2); % deg/sec/sec
end