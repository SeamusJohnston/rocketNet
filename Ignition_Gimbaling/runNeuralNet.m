function [ignite, phi_dot_dot] = runNeuralNet(weights, biases, neurons, state, geometry, NN, sim)
	inputs = [state.phi_dot, ...
             state.phi_dot_dot, ...
             state.theta_dot, ...
             state.vel_x, ...
             state.vel_z, ...
             state.theta, ...
             state.phi, ...
             state.pos_cg(1)-geometry.pos_pad(1), ...
             state.pos_cg(3)-geometry.pos_pad(3)];

neurons.one = inputs * weights.one(:,:,sim.run) - biases.one(:,sim.run).';
% Z = neurons.one > 0;
% neurons.one = Z .* neurons.one;

if NN.enable_HL2
    neurons.two = neurons.one * weights.two(:,:,sim.run) - biases.two(:,sim.run).'; 
    Z = neurons.two > 0;
    neurons.two = Z .* neurons.two;
end

if NN.enable_HL3
    neurons.three = neurons.two * weights.three(:,:,sim.run) - biases.three(:,sim.run).';
%     Z = neurons.three > 0;
%     neurons.three = Z .* neurons.three;
end

% calculate output layer neurons
if NN.enable_HL3
    neurons.out = neurons.three * weights.out(:,:,sim.run) - biases.out(:,sim.run).';
elseif NN.enable_HL2
    neurons.out = neurons.two * weights.out(:,:,sim.run) - biases.out(:,sim.run).';
else
    neurons.out = neurons.one * weights.out(:,:,sim.run) - biases.out(:,sim.run).';
end


ignite      = neurons.out(1); % output(1); % m
phi_dot_dot = neurons.out(2); % output(2); % deg/sec/sec
end