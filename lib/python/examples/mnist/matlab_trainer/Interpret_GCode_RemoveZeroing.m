% clear all
close all
clc

%% Load G code data
filename = filename_gval; % 'xyzCalibration_cube';
load(['gcode_data_',filename,'.mat']) 

Remove_Zeroing = 1;
Feedrate_edit = 1; % 0 for keeping the default maximum feedrate of 150 mm/s; 1 for alternative the maximum feedrate.
F_new = feedrate_gval % 100; % Feedrate setting [mm/s]

%% Data preparation
gcode_size = length(X); % can pick how many blocks are loaded. Use length(X) to load the entire code

mode = 1; % (1) = absolute positioning, (0) = relative positioning
for i = 1:length(G)
    if G(i) == 91 % if relative positinoing
        mode = 0;
    end
    
    if G(i) == 90 % if absolute positinoing
        mode = 1;
    end
    
    if G(i) == 28 % if absolute positinoing
        mode = 1;
    end
    
    if mode < 0.5
        X(i) = X(i) + X(i-1);
        Y(i) = Y(i) + Y(i-1);
        Z(i) = Z(i) + Z(i-1);
        E(i) = E(i) + E(i-1);
    end
end

G92_index = find(G == 92); % reset extruder length
for j = 1:length(G92_index)
    G92_pos = G92_index(length(G92_index)-j+1);
    E(G92_pos:end) = E(G92_pos:end) + E(G92_pos-1);
end

Px = X(1:gcode_size); % key points in x [mm]
Py = Y(1:gcode_size); % key points in y [mm]
Pz = Z(1:gcode_size); % key points in z [mm]
Pe = E(1:gcode_size); % key points in e [mm]
error_start = find(F == max(F),1,'first');
error_end = find(F == max(F),1,'last');
F = F/60; % convert feedrate into [mm/s]
F_max = max(F);
if Feedrate_edit>0
    for idx = error_start:error_end
        if F(idx) == F_max
            F(idx) = F_new;
        end
    end
    F_max = F_new;
end

if Remove_Zeroing == 1
    % Remove printer zeroing section
    GXYZEF = [G, X, Y, Z, E, F];
    gcode_start = find( (Px~=0) + (Py~=0) + (Pz~=0), 1 );
    Px(gcode_start:gcode_size) = Px(gcode_start:gcode_size) - Px(gcode_start);
    Py(gcode_start:gcode_size) = Py(gcode_start:gcode_size) - Py(gcode_start);
end

% linear moves only
theta(1) = atan2((Py(1) - 0),(Px(1) - 0));
dz(1) = Pz(1);
de(1) = Pe(1);
for i = 2:length(Px)
    theta(i,1) = atan2((Py(i) - Py(i-1)),(Px(i) - Px(i-1))); % angle between trajectories represented by blocks (between -pi and pi)
    dz(i,1) = Pz(i) - Pz(i-1);
    de(i,1) = Pe(i) - Pe(i-1);
    if theta(i) < 0
        theta(i) = 2*pi + theta(i); % convert angle to be between 0 and 2pi
    end
end
TotalNumBlock = length(Px);

%% Convert G code data to discrete position
Ts2 = 1/1e3;
acc = acceleration_gval; % 10000; % x and y acceleration [mm/s^2]
tf_est = 2000; % Estimated finish time [THIS NEEDS TO BE CHANGED]
sim('Trajectory_Generation_from_Gcode.slx')

%% Extract moving portion and calculate velocity & acceleration
X_last = find(GCode_X_ref ~= GCode_X_ref(end),1,'last');
Y_last = find(GCode_Y_ref ~= GCode_Y_ref(end),1,'last');
moving_portion = 1:max(X_last,Y_last);

Time = 0: Ts2 : (length(moving_portion)-1)*Ts2;
GCode_X_ref = GCode_X_ref(moving_portion);
GCode_Y_ref = GCode_Y_ref(moving_portion);
GCode_Z_ref = GCode_Z_ref(moving_portion);
GCode_E_ref = GCode_E_ref(moving_portion);
z_e_start = Pz(error_start);
z_e_end = Pz(error_end);

save(['Discrete_Position_',filename,'_V_',num2str(F_max),'_A_',num2str(acc),'.mat'],'Time',...
    'GCode_X_ref','GCode_Y_ref','GCode_Z_ref','GCode_E_ref','theta','z_e_start','z_e_end');
% fileID = fopen(['Discrete_Position_',filename,'_V_',num2str(max(F)),'_A_',num2str(acc),'.txt'], 'w');
% 
% fprintf(fileID, 't X Y Z E\n');
% for i = 1:length(Time)
%     fprintf(fileID, '%d %d %f %f %f %f\n', Time(i), GCode_X_ref(i), GCode_Y_ref(i), GCode_Z_ref(i), GCode_E_ref(i));
% end
% 
% fclose(fileID);

GCode_VX_ref = gradient(GCode_X_ref, Ts2); % velocity of the X-axis trajecotry
GCode_VY_ref = gradient(GCode_Y_ref, Ts2); % velocity of the Y-axis trajecotry
GCode_AX_ref = gradient(GCode_VX_ref,Ts2); % acceleration of the X-axis trajecotry
GCode_AY_ref = gradient(GCode_VY_ref,Ts2); % acceleration of the Y-axis trajecotry

%% Plot for visualization
% figure;
% plot3(GCode_X_ref, GCode_Y_ref, GCode_Z_ref); 
% axis equal;

% figure;
% subplot(321); plot(GCode_X_ref);  ax1 = gca;
% subplot(322); plot(GCode_Y_ref);  ax2 = gca;
% subplot(323); plot(GCode_VX_ref); ax3 = gca;
% subplot(324); plot(GCode_VY_ref); ax4 = gca;
% subplot(325); plot(GCode_AX_ref); ax5 = gca;
% subplot(326); plot(GCode_AY_ref); ax6 = gca;
% linkaxes([ax1,ax2,ax3,ax4,ax5,ax6],'x');

