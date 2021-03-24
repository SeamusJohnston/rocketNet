function plotResult(x, dt, speedFactor)   
    close all;
    figure;
    
    % Initialize
    center = [x(1,1); x(2,1)];
    theta = pi/2-x(5,1);
    height = 0.1;
    width = 0.5;

    % Initialize plot with starting position
    coords = [center(1)-(width/2) center(1)-(width/2) center(1)+(width/2)  center(1)+(width/2);...
      center(2)-(height/2) center(2)+(height/2) center(2)+(height/2)  center(2)-(height/2)];
    R = [cos(theta) sin(theta);...
        -sin(theta) cos(theta)];
    rot_coords = R*(coords-repmat(center,[1 4]))+repmat(center,[1 4]);
    p = patch('XData', rot_coords(1,:), 'YData', rot_coords(2,:),'FaceColor', [0 0 1]);
    
    % Setup plot limits
    xlim([-1+min(x(1,:)) 1+max(x(1,:))]);
    ylim([0 x(2,1)+1]);
    axis equal;
    grid on;
    
    for i=1:length(x(1,:))
        center = [x(1,i); x(2,i)];
        theta = pi/2-x(5,i);
        height = 0.1;
        width = 0.5;
        
        % Determine vectex location
        coords = [center(1)-(width/2) center(1)-(width/2) center(1)+(width/2)  center(1)+(width/2);...
          center(2)-(height/2) center(2)+(height/2) center(2)+(height/2)  center(2)-(height/2)];
        R = [cos(theta) sin(theta);...
            -sin(theta) cos(theta)];
        rot_coords = R*(coords-repmat(center,[1 4]))+repmat(center,[1 4]);
        
        % Update plot data
        p.XData = rot_coords(1,:);
        p.YData = rot_coords(2,:);
        
        % Redraw
        drawnow
        pause(dt/speedFactor);
    end
end