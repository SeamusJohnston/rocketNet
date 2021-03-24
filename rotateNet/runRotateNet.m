function runRotateNet(viz)
    % Train the network off of simulation data
    
    % PARAMETERS
    fin_angle = 0; % Initial fin angle (0 is fully extended), rad
    v_t = 5; % Terminal velocity (m/s)
    t_sim = 10; % s Max time (in case rocket doesn't hit the ground)
    dt = 0.01; % s
    h = 40; % start height
    
    % Initialize dynamics model
    x_init = [0; h; 0; -v_t; pi/2; 0;];
    rocket = rocketModel(v_t, fin_angle, x_init);
    
    % Initialize network
    learnRate = 0.95;
    epsilon = 0.9; % How much we want to explore
    eps_decay = 0.05; % epsilon decay percent every iteration
    net = rotateNet(learnRate, 2, 24, epsilon, eps_decay);
        
    % Run the simulation
    x = x_init;
    while (rocket.t <= t_sim && ~rocket.impact)
        
        % Calculate inputs from net
        u = net.determineAction(rocket.x);
        
        % Simulate dynamics
        rocket = rocket.stepDynamics(u, dt);
        rocket = rocket.checkForImpact();
        
        % Learn
        net = net.decay();
        
        % Save output
        x = [x rocket.x];
    end
    
    % Visualize sim results
    if viz
        plotResult(x, dt, 1);
    end
    
    % Score the results
    score = evaluateScore(x)
    
    % Backpropagation
    
    
end

% best score is 0
function score = evaluateScore(x)
    % Score is based off of landing at 0 degrees
    score = -(abs(x(5)) + 0.25*abs(x(6)))^2;
end