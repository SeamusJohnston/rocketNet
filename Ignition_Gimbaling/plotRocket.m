function [] = plotRocket(state, geometry, sim)
p1(1) = state.pos_cg(1)-sind(state.theta)*geometry.l_cg;
p1(3) = state.pos_cg(3)-cosd(state.theta)*geometry.l_cg;
p0(1) = p1(1) + sind(state.theta)*geometry.l_rocket;
p0(3) = p1(3) + cosd(state.theta)*geometry.l_rocket;
if state.motorIsBurning
    p2(1) = p1(1) - sind(state.theta+state.phi)*geometry.l_plume;
    p2(3) = p1(3) - cosd(state.theta+state.phi)*geometry.l_plume;
else
    p2(1) = p1(1);
    p2(3) = p1(3);
end
rocketx = [p0(1) p1(1) p2(1)];
rocketz = [p0(3) p1(3) p2(3)];
plot(rocketx, rocketz, 'o-', geometry.pos_pad(1),geometry.pos_pad(3), 'x')
axis (gca, 'equal');
axis ([-1 25 -5 55]);
pause(sim.delay)
end