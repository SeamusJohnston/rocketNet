%clear
clc
format compact

% Simulation Parameters
sim       =      struct('simBestRuns', 1, ...
                        'topTen' , 10 , ... % sim the top ten results
                        'steps' , 75, ...
                        'run', 1, ...
                        'doPlot', 0, ...
                        'delay', .01, ...
                        'timePerStep' , .05); % sec, above .05 sim seems inconsistant 

% Neural Net, up to three hidden layers
NN        =      struct('isMutating' , false, ...
                        'runsPerGeneration' , 100000, ...
                        'num_inputs' , 8, ...
                        'num_outputs', 2, ...
                        'num_neurons_HL1' , 12, ...
                        'enable_HL2' , true, ... % TODO ensure false if HL3 if false 
                        'num_neurons_HL2' , 12, ...
                        'enable_HL3' , false, ...
                        'num_neurons_HL3' , 3);

% Physics Properties
physics     =    struct('Iyy_cg' , 0.2, ...        % kg*m^2, mass moment of inertia
                        'm_rocket' , 2.0, ...     % kg
                        'g' , 9.81, ...           % m/s/s
                        'thrust' , 30, ...        % Newtons
                        't_burn' , 3, ...         % sec
                        'gimbalSpeedLimit' , 200);% deg/sec

% Geometry
geometry   =     struct('l_cg'    , .5, ...       % m, distance from gimbal to cg
                        'l_rocket' , 2, ...       % m, length of rocket
                        'l_plume'  , .6,...       % m, length of plume
                        'pos_pad'  , [15, 0, 0],...% m, x,y,z position of pad
                        'max_gimbal_angle' , 50); % deg

% Initialize State, TODO make monte carlo function
state       =   struct(...
                        'pos_cg' , [12, 0, 40], ...% m, x,y,z initial pos of cg
                        'vel_x' ,  0, ...         % m/s
                        'vel_z' , -10, ...        % m/s
                        'theta' , 0, ...          % deg, angle of rocket, vertical is zero
                        'theta_dot' , 0, ...      % deg/s, aka w_rocket
                        'theta_dot_dot' , 0, ...  % deg/s/s
                        'phi' , 0, ...            % deg, angle of gimble
                        'phi_dot' , 0, ...        % deg/s, aka w_gimbal
                        'phi_dot_dot' , 0, ...    % deg/s/s
                        'ignite' , -1, ...        % ignite when value is positive
                        't_burning' , 0, ...      % time motor has been burning
                        'motorIsBurning' , false, ...
                        'motorBurnedOut' , false);

% Initialize the neural net, TODO: add importing NN
if ~NN.isMutating && ~sim.simBestRuns
    [weights, biases] = initializeWeightsBiases(NN);
end

if sim.simBestRuns
    [scoreSorted,indexs] = sort(score);
    sim.doPlot = 1;
    for i = 1:sim.topTen
        sim.run = indexs(i);
        [runScore] = runSim(weights, biases, neurons, sim, geometry, state, NN, physics);
    end
else
    [score, neurons] = initializeScoresNeurons(NN);
    for run = 1:NN.runsPerGeneration
        sim.run = run;
        [runScore] = runSim(weights, biases, neurons, sim, geometry, state, NN, physics);
        score(run) = runScore;
    end
    %sort and save the best 1000ish
    %save('12_12_4Mil.mat', '-v7.3')
end

