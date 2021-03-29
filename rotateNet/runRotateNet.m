function runRotateNet(viz, num_episodes)
    % Train the network off of simulation data
    
    % PARAMETERS
    fin_angle = 0; % Initial fin angle (0 is fully extended), rad
    v_t = 5; % Terminal velocity (m/s)
    t_sim = 10; % s Max time (in case rocket doesn't hit the ground)
    dt = 0.01; % s
    h = 40; % start height
    
    % Initialize dynamics model
    x_init = [0; h; 0; -v_t; pi/2; 0;];
    scores_on_impact = []; % scores on impact
    x_best = []; % best run
    rocket = rocketModel(v_t, fin_angle, x_init);
    
    % Initialize network
    learnRate = 0.95;
    epsilon = 0.9; % How much we want to explore
    eps_decay = 0.05; % epsilon decay percent every iteration
    net = rotateNet(learnRate, 2, 24, epsilon, eps_decay);
    
    for i = 1:num_episodes
        % Run the simulation
        x = x_init;
        while (rocket.t <= t_sim && ~rocket.impact)

            % Calculate inputs from net
            [net, u] = net.determineAction(rocket.x);

            % Simulate dynamics
            prev_state = rocket.x;
            rocket = rocket.stepDynamics(u, dt);
            rocket = rocket.checkForImpact();

            % Score the results
            score = evaluateScore(rocket.x);
            if rocket.impact
                scores_on_impact = [scores_on_impact score];
            end

            % Learn
            net = net.remember(rocket.impact, u, rocket.x, prev_state, score);
            net = net.experience_replay(20, rocket, dt);

            % Save output
            x = [x rocket.x];
            if score == max(scores_on_impact)
                x_best = x;
            end
        end
    end
    
    % Visualize sim results
    if viz
        plotResult(x_best, dt, 1);
    end
end

% best score is 0
function score = evaluateScore(x)
    % Score is based off of landing at 0 degrees
    score = -(abs(x(5)) + 0.25*abs(x(6)))^2;
end