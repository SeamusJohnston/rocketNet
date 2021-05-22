% v2.2 x=3 772.1558 735.8311 708 691 680 678 sudden jump 349 260 241 236.4421
% 188.95 188.8424 183.5263 178.9500 170.5653 166.7476 stalled 166.6415 
% x=2 120.6411 118.67
%x=3 6 6 106.3154 95.7954 85.86 85.58 84.64
%x 3 4 4 139.24
%x=3 8 8 300
%x=3 6 6 123.06 110.39 105 89 80 79.39 75.45 72.2821 70.32 68.87 66.29
%x=3 11 6 65.96 65.2611 65.24 64.47 63.99 62.87 62.27 60.45 58.36 57.06
% mcs 1.1/.23 55.67 52.59 51.71 50.57 49.57 47.96 46.82 45.85 42.88
%four more 78.09 76.64

clc
format compact
StopLoopinG = 0;
    %save('v2.3.mat', '-v7.3')
x = 3;
% 1. Make a ton of guess with 1 constant initial state (~100k)
% 4. Do 5 initial pos x 3 theta_dots, keep same for whole generation
% 5. Randomize all initial states, keep same for whole generation
% 6. Add noise to physiscs and train

% Simulation Parameters
sim       =      struct('steps' , 500, ... % 75 work for training
                        'run', 1, ...
                        'generation', 1 , ... in the family sense
                        'doPlot', 0, ...
                        'xposVariance', x, ... m, max variance for initial position when plotting
                        'zposVariance', x, ... 
                        'theta_dotVar', x, ... deg/s max variance for initial theta_dot
                        'delay', .01, ...
                        'timePerStep' , .04); % sec, above .05 sim seems inconsistant 

% Neural Net, up to three hidden layers
NN        =      struct('isTraining' , 1, ... % false triggers initial guesses
                        'goWeightByWeight', 0, ... % adjust one weight and bias at a time
                        'doMixUpInitialState' , 1 , ... % Turn on for step 4
                        'runsPerGeneration' , 200, ... % First training round will run all
                        'breakIfBeat' , false , ... % finish the generation for a new high score
                        'numGenerations', 1 , ...
                        'numInitialGuesses' , 10000, ...
                        'winningIndex', 0, ... % set once per generation
                        'winningScore', 0, ... 
                        'secondPlace', 0, ... % second place score
                        'mutationCoef', .01 , ... % multiplier for random change
                        'mutationReducer', .5 , ... % % reduction of mutationCoef after success
                        'num_inputs' , 9, ...
                        'num_outputs', 2, ...
                        'num_neurons_HL1' , 14, ...
                        'enable_HL2' , true, ... % TODO ensure false if HL3 if false 
                        'num_neurons_HL2' , 6, ...
                        'enable_HL3' , false, ...
                        'num_neurons_HL3' , 6);

% Physics Properties
physics     =    struct('Iyy_cg' , 0.1, ...        % kg*m^2, mass moment of inertia
                        'm_rocket' , 2.0, ...     % kg
                        'g' , 9.81, ...           % m/s/s
                        'thrust' , 30, ...        % Newtons
                        't_burn' , 3, ...         % sec
                        'gimbalSpeedLimit' , 200);% deg/sec

% Geometry
geometry   =     struct('l_cg'    , .5, ...       % m, distance from gimbal to cg
                        'l_rocket' , 2, ...       % m, length of rocket
                        'l_plume'  , .9,...       % m, length of plume
                        'pos_pad'  , [12, 0, 0],...% m, x,y,z position of pad
                        'max_gimbal_angle' , 50); % deg

% Initialize State, TODO make monte carlo function
state       =   struct(...
                        'pos_cg' , [12, 0, 40], ...% m, x,y,z initial pos of cg [12, 0, 40]
                        'vel_x' ,  0, ...         % m/s
                        'vel_z' , -10, ...        % m/s
                        'theta' , 0, ...          % deg, angle of rocket, vertical is zero
                        'theta_dot' , 0 , ...      % deg/s, aka w_rocket
                        'theta_dot_dot' , 0, ...  % deg/s/s
                        'phi' , 0, ...            % deg, angle of gimble
                        'phi_dot' , 0, ...        % deg/s, aka w_gimbal
                        'phi_dot_dot' , 0, ...    % deg/s/s
                        'ignite' , -1, ...        % ignite when value is positive
                        't_burning' , 0, ...      % time motor has been burning
                        'motorIsBurning' , false, ...
                        'motorBurnedOut' , false);
counter = 0;
nomInitPos = state.pos_cg;
nomTheta_dot = state.theta_dot;
% 4. Do 5 initial pos x 3 theta_dots, keep same for whole generation
initRotation = [nomTheta_dot - sim.theta_dotVar, nomTheta_dot, nomTheta_dot + sim.theta_dotVar];
while(~StopLoopinG)
    
    if counter >= 1
        NN.mutationCoef = 1000;
        NN.goWeightByWeight = true; 
        if counter >= 2
            NN.mutationCoef = .00001;
            NN.goWeightByWeight = false;
            counter = 0;
        end
    end
    counter = counter + 1;
    NN.goWeightByWeight
    % 1. Make a ton of guesses with 1 constant initial state
    if ~NN.isTraining
        sim.generation = 1;
        [weights, biases] = initializeWeightsBiases(NN,[],[]);
        [score, neurons] = initializeScoresNeurons(NN);
        for run = 1:NN.numInitialGuesses
            if StopLoopinG == 1 
                break
            end
            sim.run = run;
            [runScore] = runSim(weights, biases, neurons, sim, geometry, state, NN, physics);
            score(run) = runScore;
        end
        % Sort scores from lowest (good) to highest (bad)
        [scoreSorted,scoreIndexs] = sort(score);
        winningIndex = scoreIndexs(1);
        winningScore = scoreSorted(1);
        StopLoopinG = 1;
    elseif NN.goWeightByWeight
        NN.winningIndex = winningIndex;
        NN.winningScore = winningScore;
        sim.run = NN.winningIndex;
        for gen = 1:NN.numGenerations
            changes = 0;
            sim.generation = gen;
            if StopLoopinG == 1 
                break
            end
%             for i = 1:NN.num_neurons_HL2
%                 for n = 1:NN.num_neurons_HL3 
%                     winner = weights.three(i,n,NN.winningIndex);
%                     weights.three(i,n,NN.winningIndex) = winner * (1 + NN.mutationCoef*rand);
%                     simSweepScore = simSweep(weights, biases, neurons, sim, geometry, state, NN, physics, initRotation, nomInitPos);
%                     if simSweepScore < NN.winningScore
%                         NN.winningScore = simSweepScore;
%                         changes = changes + 1;
%                     else
%                         weights.three(i,n,NN.winningIndex) = winner * (1 - NN.mutationCoef*rand);
%                         simSweepScore = simSweep(weights, biases, neurons, sim, geometry, state, NN, physics, initRotation, nomInitPos);
%                         if simSweepScore < NN.winningScore
%                             NN.winningScore = simSweepScore;
%                             changes = changes + 1;
%                         else
%                             weights.three(i,n,NN.winningIndex) = winner;
%                         end
%                     end
%                 end
%                 %Print status
%                 clc
%                 HL2_neuron = i
%                 changes
%                 winningScore = NN.winningScore
%             end
            % biases three
%             for n = 1:NN.num_neurons_HL3 
%                 winner = biases.three(n,NN.winningIndex);
%                 biases.three(n,NN.winningIndex) = winner * (1 + NN.mutationCoef*rand);
%                 simSweepScore = simSweep(weights, biases, neurons, sim, geometry, state, NN, physics, initRotation, nomInitPos);
%                 if simSweepScore < NN.winningScore
%                     NN.winningScore = simSweepScore;
%                     changes = changes + 1;
%                 else
%                     biases.three(n,NN.winningIndex) = winner * (1 - NN.mutationCoef*rand);
%                     simSweepScore = simSweep(weights, biases, neurons, sim, geometry, state, NN, physics, initRotation, nomInitPos);
%                     if simSweepScore < NN.winningScore
%                         NN.winningScore = simSweepScore;
%                         changes = changes + 1;
%                     else
%                         biases.three(n,NN.winningIndex) = winner;
%                     end
%                 end
%             end
%             %Print status
%             clc
%             biases_three = n
%             changes
%             winningScore = NN.winningScore

            
            % biases out
            for n = 1:NN.num_outputs 
                winner = biases.out(n,NN.winningIndex);
                biases.out(n,NN.winningIndex) = winner * (1 + NN.mutationCoef*rand);
                simSweepScore = simSweep(weights, biases, neurons, sim, geometry, state, NN, physics, initRotation, nomInitPos);
                if simSweepScore < NN.winningScore
                    NN.winningScore = simSweepScore;
                    changes = changes + 1;
                else
                    biases.out(n,NN.winningIndex) = winner * (1 - NN.mutationCoef*rand);
                    simSweepScore = simSweep(weights, biases, neurons, sim, geometry, state, NN, physics, initRotation, nomInitPos);
                    if simSweepScore < NN.winningScore
                        NN.winningScore = simSweepScore;
                        changes = changes + 1;
                    else
                        biases.out(n,NN.winningIndex) = winner;
                    end
                end
            end
            %Print status
            clc
            biases_out = n
            changes
            winningScore = NN.winningScore
            
            for i = 1:NN.num_inputs
                for n = 1:NN.num_neurons_HL1 
                    winner = weights.one(i,n,NN.winningIndex);
                    weights.one(i,n,NN.winningIndex) = winner * (1 + NN.mutationCoef*rand);
                    simSweepScore = simSweep(weights, biases, neurons, sim, geometry, state, NN, physics, initRotation, nomInitPos);
                    if simSweepScore < NN.winningScore
                        NN.winningScore = simSweepScore;
                        changes = changes + 1;
                    else
                        weights.one(i,n,NN.winningIndex) = winner * (1 - NN.mutationCoef*rand);
                        simSweepScore = simSweep(weights, biases, neurons, sim, geometry, state, NN, physics, initRotation, nomInitPos);
                        if simSweepScore < NN.winningScore
                            NN.winningScore = simSweepScore;
                            changes = changes + 1;
                        else
                            weights.one(i,n,NN.winningIndex) = winner;
                        end
                    end
                end
                %Print status
                clc
                input = i
                changes
                winningScore = NN.winningScore
            end
            % biases one
            for n = 1:NN.num_neurons_HL1 
                winner = biases.one(n,NN.winningIndex);
                biases.one(n,NN.winningIndex) = winner * (1 + NN.mutationCoef*rand);
                simSweepScore = simSweep(weights, biases, neurons, sim, geometry, state, NN, physics, initRotation, nomInitPos);
                if simSweepScore < NN.winningScore
                    NN.winningScore = simSweepScore;
                    changes = changes + 1;
                else
                    biases.one(n,NN.winningIndex) = winner * (1 - NN.mutationCoef*rand);
                    simSweepScore = simSweep(weights, biases, neurons, sim, geometry, state, NN, physics, initRotation, nomInitPos);
                    if simSweepScore < NN.winningScore
                        NN.winningScore = simSweepScore;
                        changes = changes + 1;
                    else
                        biases.one(n,NN.winningIndex) = winner;
                    end
                end
            end
            %Print status
            clc
            biases_one = n
            changes
            winningScore = NN.winningScore

            for i = 1:NN.num_neurons_HL1
                for n = 1:NN.num_neurons_HL2 % MUST CHANGE just next layer 
                    winner = weights.two(i,n,NN.winningIndex); %same
                    weights.two(i,n,NN.winningIndex) = winner * (1 + NN.mutationCoef*rand);
                    simSweepScore = simSweep(weights, biases, neurons, sim, geometry, state, NN, physics, initRotation, nomInitPos);
                    if simSweepScore < NN.winningScore
                        NN.winningScore = simSweepScore;
                        changes = changes + 1;
                    else
                        weights.two(i,n,NN.winningIndex) = winner * (1 - NN.mutationCoef*rand);
                        simSweepScore = simSweep(weights, biases, neurons, sim, geometry, state, NN, physics, initRotation, nomInitPos);
                        if simSweepScore < NN.winningScore
                            NN.winningScore = simSweepScore;
                            changes = changes + 1;
                        else
                            weights.two(i,n,NN.winningIndex) = winner;
                        end
                    end
                end
                %Print status
                clc
                HL1_neuron = i
                changes
                winningScore = NN.winningScore
            end
            % biases two
            for n = 1:NN.num_neurons_HL2 
                winner = biases.two(n,NN.winningIndex);
                biases.two(n,NN.winningIndex) = winner * (1 + NN.mutationCoef*rand);
                simSweepScore = simSweep(weights, biases, neurons, sim, geometry, state, NN, physics, initRotation, nomInitPos);
                if simSweepScore < NN.winningScore
                    NN.winningScore = simSweepScore;
                    changes = changes + 1;
                else
                    biases.two(n,NN.winningIndex) = winner * (1 - NN.mutationCoef*rand);
                    simSweepScore = simSweep(weights, biases, neurons, sim, geometry, state, NN, physics, initRotation, nomInitPos);
                    if simSweepScore < NN.winningScore
                        NN.winningScore = simSweepScore;
                        changes = changes + 1;
                    else
                        biases.two(n,NN.winningIndex) = winner;
                    end
                end
            end
            %Print status
            clc
            biases_two = n
            changes
            winningScore = NN.winningScore
            
            for i = 1:NN.num_neurons_HL2
                for n = 1:NN.num_outputs 
                    winner = weights.out(i,n,NN.winningIndex);
                    weights.out(i,n,NN.winningIndex) = winner * (1 + NN.mutationCoef*rand);
                    simSweepScore = simSweep(weights, biases, neurons, sim, geometry, state, NN, physics, initRotation, nomInitPos);
                    if simSweepScore < NN.winningScore
                        NN.winningScore = simSweepScore;
                        changes = changes + 1;
                    else
                        weights.out(i,n,NN.winningIndex) = winner * (1 - NN.mutationCoef*rand);
                        simSweepScore = simSweep(weights, biases, neurons, sim, geometry, state, NN, physics, initRotation, nomInitPos);
                        if simSweepScore < NN.winningScore
                            NN.winningScore = simSweepScore;
                            changes = changes + 1;
                        else
                            weights.out(i,n,NN.winningIndex) = winner;
                        end
                    end
                end
                %Print status
                clc
                HL2_neuron = i
                changes
                winningScore = NN.winningScore
            end
           
            if changes == 0
                NN.mutationCoef = NN.mutationCoef*NN.mutationReducer;
            end
            if winningScore < highScore
                highScore = winningScore;
                save('v2.3.mat', '-v7.3')
                pause(1)
            end
            %sim
            sim.doPlot = true;
            state.theta_dot = nomTheta_dot + 2*(rand-.5)*sim.theta_dotVar;
            state.pos_cg = nomInitPos + [2*(rand-.5)*sim.xposVariance, 0, 2*(rand-.5)*sim.zposVariance];
            [runScore] = runSim(weights, biases, neurons, sim, geometry, state, NN, physics);
            sim.doPlot = false;
        end
    elseif NN.isTraining
        for gen = 1:NN.numGenerations
            sim.generation = gen;
            if StopLoopinG == 1 
                break
            end
            NN.winningIndex = winningIndex;
            NN.winningScore = winningScore;
            [weights, biases] = initializeWeightsBiases(NN, weights, biases);
            [score, neurons] = initializeScoresNeurons(NN);
            for run = 1:NN.runsPerGeneration
                sim.run = run;
                for i = 1 : length(initRotation)
                    state.theta_dot = initRotation(i);
                    % nominal position
                    state.pos_cg = nomInitPos;
                    [runScore] = runSim(weights, biases, neurons, sim, geometry, state, NN, physics);
                    score(run) = score(run) + runScore;
                    % plus four corners
                    state.pos_cg = nomInitPos + [sim.xposVariance 0 sim.zposVariance];
                    [runScore] = runSim(weights, biases, neurons, sim, geometry, state, NN, physics);
                    score(run) = score(run) + runScore;
                    state.pos_cg = nomInitPos + [-sim.xposVariance 0 sim.zposVariance];
                    [runScore] = runSim(weights, biases, neurons, sim, geometry, state, NN, physics);
                    score(run) = score(run) + runScore;
                    state.pos_cg = nomInitPos + [sim.xposVariance 0 -sim.zposVariance];
                    [runScore] = runSim(weights, biases, neurons, sim, geometry, state, NN, physics);
                    score(run) = score(run) + runScore;
                    state.pos_cg = nomInitPos + [-sim.xposVariance 0 -sim.zposVariance];
                    [runScore] = runSim(weights, biases, neurons, sim, geometry, state, NN, physics);
                    score(run) = score(run) + runScore;
                    % plus four more
                    state.pos_cg = nomInitPos + [sim.xposVariance/2 0 sim.zposVariance/2];
                    [runScore] = runSim(weights, biases, neurons, sim, geometry, state, NN, physics);
                    score(run) = score(run) + runScore;
                    state.pos_cg = nomInitPos + [-sim.xposVariance/2 0 sim.zposVariance/2];
                    [runScore] = runSim(weights, biases, neurons, sim, geometry, state, NN, physics);
                    score(run) = score(run) + runScore;
                    state.pos_cg = nomInitPos + [sim.xposVariance/2 0 -sim.zposVariance/2];
                    [runScore] = runSim(weights, biases, neurons, sim, geometry, state, NN, physics);
                    score(run) = score(run) + runScore;
                    state.pos_cg = nomInitPos + [-sim.xposVariance/2 0 -sim.zposVariance/2];
                    [runScore] = runSim(weights, biases, neurons, sim, geometry, state, NN, physics);
                    score(run) = score(run) + runScore;
                end
                % end generation for a winning score
                if NN.breakIfBeat == true && score(run) < NN.winningScore
                    break
                end
            end
            [scoreSorted,scoreIndexs] = sort(score(score > 0));
            NN.winningIndex = scoreIndexs(1);
            NN.winningScore = scoreSorted(1);
            winningIndex = NN.winningIndex;
            winningScore = NN.winningScore;
            if winningScore < highScore
                highScore = winningScore;
                save('v2.3.mat', '-v7.3')
                pause(1)
            end
            NN.secondPlace = scoreSorted(2);
            if NN.winningIndex == 1
                NN.mutationCoef = NN.mutationCoef*NN.mutationReducer;
            end
            % Plot the winner of each generation from a random initial position
            sim.doPlot = true;
            sim.run = NN.winningIndex;
            state.theta_dot = nomTheta_dot + 2*(rand-.5)*sim.theta_dotVar;
            state.pos_cg = nomInitPos + [2*(rand-.5)*sim.xposVariance, 0, 2*(rand-.5)*sim.zposVariance];
            [runScore] = runSim(weights, biases, neurons, sim, geometry, state, NN, physics);
            sim.doPlot = false;
        end
    else
        error('Error')
    end
end
