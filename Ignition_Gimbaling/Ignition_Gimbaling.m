%clear score 190
clc
format short
StopLoopinG = 0;
    %save('V1_4.mat', '-v7.3')

% 1. Make a ton of guess with 1 constant initial state (~100k)
% 2. Evaluate top ~ 0.01% with varied initial state, accumulate scores, pick winner 
% 3. Mutate winner over X generations w/ 1 initial state (~1k runs 60-3k gens)
%       continuously decrease mutation coefficient
% 4. Do ~3 variations of an initial state (i.e. initial pos), keep same for whole generation
%       repeat for each initial state param
% 5. Vary all initial states, keep same for whole generation
% 6. Add noise to physiscs and train

% Simulation Parameters
sim       =      struct('simBestRuns', 0, ...
                        'topTen' , 5 , ... % sim the top ten results
                        'steps' , 500, ... % 75 work for training
                        'run', 1, ...
                        'generation', 1 , ... in the family sense
                        'doPlot', 0, ...
                        'xposVariance', 10, ... m, max variance for initial position when plotting
                        'zposVariance', 6, ... 
                        'delay', .01, ...
                        'timePerStep' , .04); % sec, above .05 sim seems inconsistant 

% Neural Net, up to three hidden layers
NN        =      struct('isTraining' , 1, ... % false triggers initial guesses
                        'doMixUpInitialState' , 1 , ... Turn on for step 4
                        'numVariations' , 3 , ... 
                        'runsPerGeneration' , 25, ...
                        'numGenerations', 10000 , ...
                        'numInitialGuesses' , 50000, ...
                        'numInitialCandidates', 10, ... num initial winners for step 2
                        'winningIndex', 0, ... % set once per generation
                        'winningScore', 0, ... 
                        'mutationCoef', .02078 , ... % multiplier for random change
                        'mutationReducer', 1 , ... % % reduction of mutationCoef after success
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
                        'l_plume'  , .9,...       % m, length of plume
                        'pos_pad'  , [12, 0, 0],...% m, x,y,z position of pad
                        'max_gimbal_angle' , 50); % deg

% Initialize State, TODO make monte carlo function
state       =   struct(...
                        'pos_cg' , [12, 0, 40], ...% m, x,y,z initial pos of cg [12, 0, 40]
                        'vel_x' ,  0, ...         % m/s
                        'vel_z' , -10, ...        % m/s
                        'theta' , 0, ...          % deg, angle of rocket, vertical is zero
                        'theta_dot' , -4, ...      % deg/s, aka w_rocket
                        'theta_dot_dot' , 0, ...  % deg/s/s
                        'phi' , 0, ...            % deg, angle of gimble
                        'phi_dot' , 0, ...        % deg/s, aka w_gimbal
                        'phi_dot_dot' , 0, ...    % deg/s/s
                        'ignite' , -1, ...        % ignite when value is positive
                        't_burning' , 0, ...      % time motor has been burning
                        'motorIsBurning' , false, ...
                        'motorBurnedOut' , false);

nomInitPos = state.pos_cg;
% Sim and plot the best scoring runs if simBestRuns is set
if sim.simBestRuns
    sim.doPlot = 1;
    for i = 1:sim.topTen
%       sim.run = scoreIndexs(delusionScoreIndexs(i)); % post delusion testing
        sim.run = scoreIndexs(i); % pre delusion testing
        [runScore] = runSim(weights, biases, neurons, sim, geometry, state, NN, physics);
    end
% 1. Make a ton of guesses with 1 constant initial state
elseif ~NN.isTraining
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
    % 2. Evaluate top ~ 0.01% with varied initial state, accumulate scores
    % (delusion check)
    delusionScore = zeros(NN.numInitialCandidates, 1, 'single');
    for run = 1 : NN.numInitialCandidates
        sim.run = scoreIndexs(run);
        state.pos_cg = nomInitPos + [0 0 1];
        [runScore] = runSim(weights, biases, neurons, sim, geometry, state, NN, physics);
        delusionScore(run) = runScore;
        state.pos_cg = nomInitPos + [0 0 -1];
        [runScore] = runSim(weights, biases, neurons, sim, geometry, state, NN, physics);
        delusionScore(run) = delusionScore(run) + runScore;
        state.pos_cg = nomInitPos + [1 0 0];
        [runScore] = runSim(weights, biases, neurons, sim, geometry, state, NN, physics);
        delusionScore(run) = delusionScore(run) + runScore;
        state.pos_cg = nomInitPos + [-1 0 0];
        [runScore] = runSim(weights, biases, neurons, sim, geometry, state, NN, physics);
        delusionScore(run) = delusionScore(run) + runScore;
    end
    [delusionScoreSorted,delusionScoreIndexs] = sort(delusionScore); 
    NN.winningIndex = scoreIndexs(delusionScoreIndexs(1));
    winningIndex = NN.winningIndex;
elseif NN.isTraining
% 3. Mutate the new winner over X generations w/ 1 initial state
    for gen = 1:NN.numGenerations
        if StopLoopinG == 1 
            break
        end
        NN.winningIndex = winningIndex;
        [weights, biases] = initializeWeightsBiases(NN, weights, biases);
        [score, neurons] = initializeScoresNeurons(NN);
        % Generate three random initial positions to be used during step 4
        initPositions = cell(1,NN.numVariations);
        for i = 1:NN.numVariations
            initPositions{i} = nomInitPos + ...
                            [sim.xposVariance*2*(rand-.5), 0, sim.zposVariance*2*(rand-.5)];
        end
        for run = 1:NN.runsPerGeneration
            sim.run = run;
            sim.generation = gen;
            [runScore] = runSim(weights, biases, neurons, sim, geometry, state, NN, physics);
            score(run) = runScore;
            if NN.doMixUpInitialState
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
                % 4. Do ~3 variations of an initial state (i.e. initial pos)
                for i = 1:NN.numVariations
                    state.pos_cg = initPositions{i};
                    [runScore] = runSim(weights, biases, neurons, sim, geometry, state, NN, physics);
                    score(run) = score(run) + runScore;
                end
            end
        end
        [scoreSorted,scoreIndexs] = sort(score);
        NN.winningScore = scoreSorted(scoreIndexs(1));
        NN.winningIndex = scoreIndexs(1);
        if NN.winningIndex == 1
            NN.mutationCoef = NN.mutationCoef*NN.mutationReducer;
        end
        winningIndex = NN.winningIndex;
        % Plot the winner of each generation from a random initial position
        sim.doPlot = true;
        sim.run = NN.winningIndex;
        state.pos_cg = initPositions{1};
        [runScore] = runSim(weights, biases, neurons, sim, geometry, state, NN, physics);
        sim.doPlot = false;
%         weights.one(5,:,NN.winningIndex)
    end
else
    error('Error')
end

