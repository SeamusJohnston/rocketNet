function [score] = runSim(weights, biases, neurons, sim, geometry, state, NN, physics)
    % record initial position and velocity vector for plotting
    initPos = state.pos_cg;
    for k = 1:sim.steps
        [output1, output2] = runNeuralNet(weights, biases, neurons, state, geometry, NN, sim);
        state.ignite = output1;
        state.phi_dot_dot = output2;

        % Angular Rate of Gimbal
        state.phi_dot = state.phi_dot + state.phi_dot_dot*sim.timePerStep;
        % Apply speed limit
        if state.phi_dot > physics.gimbalSpeedLimit state.phi_dot = physics.gimbalSpeedLimit; end
        if state.phi_dot < -physics.gimbalSpeedLimit state.phi_dot = -physics.gimbalSpeedLimit; end
        % Angle of Gimbal
        state.phi = state.phi + state.phi_dot* sim.timePerStep;
        % prevent gimbal angles that are too much
        if state.phi >= geometry.max_gimbal_angle  state.phi = geometry.max_gimbal_angle; end
        if state.phi <= -geometry.max_gimbal_angle state.phi = -geometry.max_gimbal_angle; end
        % Ignite motor if ignite goes above 0
        if state.ignite > 0 && ~state.motorBurnedOut    state.motorIsBurning = true; end
        % Turn off motor if it's been burning too long
        if state.motorIsBurning && state.t_burning > physics.t_burn 
            state.motorBurnedOut = true;
            state.motorIsBurning = false;
            % Stop the sim when motor burns out, TODO: make more elegant scoring
            break
        end
        % Time engine burn
        if state.motorIsBurning   state.t_burning = state.t_burning + sim.timePerStep; end
        % Torque = leverArm * Thrust
        if state.motorIsBurning
            thrust_arm = -sind(state.phi)*geometry.l_cg;
            torque_yy = thrust_arm * physics.thrust; % N*m
        else
            torque_yy = 0;
        end
        % Angular Acceleration of Rocket, angular_accel = torque/I
        state.theta_dot_dot = torque_yy/physics.Iyy_cg; % deg/sec/sec
        % Angular Rate of Rocket
        state.theta_dot = state.theta_dot + state.theta_dot_dot*sim.timePerStep; %deg/sec
        % Angle of Rocket
        state.theta = state.theta + state.theta_dot*sim.timePerStep; % deg

        % Linear Acceleration of Rocket Due to thrust, accel = F/m
        if state.motorIsBurning
            inert_thrust_angle = state.theta + state.phi;
            thrust_x = sind(inert_thrust_angle)*physics.thrust;
            thrust_z = cosd(inert_thrust_angle)*physics.thrust;
            accel_x = thrust_x/physics.m_rocket;
            accel_z = thrust_z/physics.m_rocket - physics.g; % apply gravity
        else
            accel_x = 0;
            accel_z = -physics.g; % apply gravity
        end
        % Velocity of rocket CG
        state.vel_x = state.vel_x + accel_x * sim.timePerStep;
        state.vel_z = state.vel_z + accel_z * sim.timePerStep;
        % Position
        state.pos_cg(1) = state.pos_cg(1) + state.vel_x * sim.timePerStep;
        state.pos_cg(3) = state.pos_cg(3) + state.vel_z * sim.timePerStep;
        % Stop Simulating if we get to far underground
        if state.pos_cg(3) < -15
            break
        end
        % Score
        vertDistanceFromPad = abs(state.pos_cg(3) - geometry.pos_pad(3));
        xDistanceFromPad = abs(state.pos_cg(1) - geometry.pos_pad(1));
        score = vertDistanceFromPad + xDistanceFromPad*4;
        score = score + abs(state.vel_x);
        score = score + abs(state.vel_z);
        score = score + abs(state.theta)*.13;
        score = score + abs(state.theta_dot)*.2;
        % Plot
        if sim.doPlot == true
           plotRocket(state, geometry, sim, initPos, NN) % TODO add %identifier of weights and biases
        end
    end
    
    % print progress
    if NN.isTraining
        if rem(sim.run, NN.runsPerGeneration/100) == 0
            clc
            percentComplete = 100*sim.run/NN.runsPerGeneration
            winningScore = NN.winningScore
        end
    else
        if rem(sim.run, NN.numInitialGuesses/100) == 0
            clc
            percentComplete = 100*sim.run/NN.numInitialGuesses
        end
    end
end
