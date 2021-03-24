classdef rocketModel
    %ROCKETMODEL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        x
        m
        l_f
        l
        h
        rho
        A_b
        A_f
        g
        A_bc
        t
        impact
        c_d
    end
    
    methods
        function obj = rocketModel(v_t, fin_angle, x_init)
            %ROCKETMODEL Construct an instance of this class
            %   Detailed explanation goes here
            obj.g = 9.81;
            obj.rho = 1.225;
            obj.m = 2;
            obj.l = 0.5;
            obj.h = 0.1;
            obj.A_b = obj.l*obj.h;
            obj.A_bc = obj.h^2;
            obj.l_f = 0.0625;
            obj.A_f = obj.l_f^2;
            obj.t = 0;
            obj.impact = false;
            obj.x = x_init;
            obj = obj.calcDragCoeff(v_t, fin_angle);
        end
        
        % assumes perpendicular entry
        % calculate from terminal velocity
        function obj = calcDragCoeff(obj, v_t, fin_angle)
            obj.c_d = (obj.m * obj.g)/(0.5 * obj.rho * v_t^2 * (obj.A_b + 2 * obj.A_f * cos(fin_angle)));
        end
        
        function obj = stepDynamics(obj, u, dt)
            % States: 1 - x
            %         2 - z
            %         3 - vx
            %         4 - vz
            %         5 - theta
            %         6 - w
        
            xPrev = obj.x;
            
            % Bound input
            ub = ones(2,1)*pi/2;
            lb = -ub;
            u = min(max(u,lb),ub);
            
            % Calculate drag forces, perp areas of fins and body
            kz = obj.rho * xPrev(4)^2 * obj.c_d / (2); % Always up
            kx = obj.rho * -1 * sign(xPrev(3)) * xPrev(3)^2 * obj.c_d / (2); % Dir opposes motion
            Iy = obj.m * (obj.l^2/12 + obj.h^2/12);

            A_side_z = obj.A_b * abs(sin(xPrev(5)));
            A_tb_z = obj.A_bc * abs(cos(xPrev(5)));
            fz_body = kz * A_side_z  + kz * A_tb_z;

            A_side_x = obj.A_b * abs(cos(xPrev(5)));
            A_tb_x = obj.A_bc * abs(sin(xPrev(5)));
            fx_body = kx * A_side_x + kx * A_tb_x;

            A_fin_a = obj.A_f * cos(u(1));
            A_fin_b = obj.A_f * cos(u(2));
            fz_fin_a = kz * A_fin_a * abs(sin(xPrev(5)));
            fz_fin_b = kz * A_fin_b * abs(sin(xPrev(5)));
            fx_fin_a = kx * A_fin_a * abs(cos(xPrev(5)));
            fx_fin_b = kx * A_fin_b * abs(cos(xPrev(5)));

            % assume drag on body doesnt affect rotation, calc moments due to fins
            dx_fin_a = (0.25 * obj.l + obj.l_f/2) * -sin(xPrev(5));
            dz_fin_a = (0.25 * obj.l + obj.l_f/2) * cos(xPrev(5));
            dx_fin_b = (0.25 * obj.l + obj.l_f/2) * sin(xPrev(5));
            dz_fin_b = (0.25 * obj.l + obj.l_f/2) * -cos(xPrev(5));
            My = dx_fin_b * fz_fin_b + dx_fin_a * fz_fin_a + dz_fin_b * fx_fin_b + dz_fin_a * fx_fin_a;

            % update states
            obj.x = xPrev + [xPrev(3); xPrev(4); (fx_body + fx_fin_a + fx_fin_b)/obj.m;
                        (fz_body + fz_fin_a + fz_fin_b)/obj.m-obj.g; xPrev(6); My/Iy;] * dt;
            obj.t = obj.t + dt;
        end
        
        function obj = checkForImpact(obj)
            center = [obj.x(1); obj.x(2)];
            theta = pi/2-obj.x(5);

            % plot rectangle
            coords = [center(1)-(obj.l/2) center(1)-(obj.l/2) center(1)+(obj.l/2)  center(1)+(obj.l/2);...
              center(2)-(obj.h/2) center(2)+(obj.h/2) center(2)+(obj.h/2)  center(2)-(obj.h/2)];
            R = [cos(theta) sin(theta);...
                -sin(theta) cos(theta)];
            rot_coords = R*(coords-repmat(center,[1 4]))+repmat(center,[1 4]);

            for i=1:length(rot_coords)
                if rot_coords(2,i) <= 0
                    obj.impact = true;
                end
            end
        end
    end
end

