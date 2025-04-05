clear all
close all
clc

%% Load trajecotry
filename = 'xyzCalibration_cube';
load(['Discrete_Position_',filename,'.mat'])

%% Load simulated system dynamics for Ender 3 Pro
load('TF_Ender3_X_axis.mat')
load('TF_Ender3_Y_axis.mat')

%% Simulate output trajectory
X_in = GCode_X_ref;
Y_in = GCode_Y_ref;
Z_in = GCode_Z_ref;
E_in = GCode_E_ref;

X_out = lsim(sys_x1, X_in, Time);
Y_out = lsim(sys_y1, Y_in, Time);
Z_out = Z_in;
E_out = E_in;

Time = reshape(Time,[],1);
Data = [Time, X_in, Y_in, Z_in, E_in, X_out, Y_out, Z_out, E_out];
writematrix(Data,['IO_Data_',filename,'.csv']);

%% Plot for visualization
figure;
subplot(121); 
plot3(X_in,  Y_in,  Z_in ); 
title('Nominal shape'); axis equal;
subplot(122); 
plot3(X_out, Y_out, Z_out);
title('Output shape'); axis equal;

figure;
subplot(221); ax1 = gca;
plot(Time, X_in); hold on;
plot(Time, X_out);
legend('Input','Output');
title('X-axis Input/Output Trajectory [mm]');
subplot(223); ax3 = gca;
plot(Time, X_out - X_in);
title('X-axis Tracking Error [mm]');
subplot(222); ax2 = gca;
plot(Time, Y_in); hold on;
plot(Time, Y_out);
legend('Input','Output');
title('Y-axis Input/Output Trajectory [mm]');
subplot(224); ax4 = gca;
plot(Time, Y_out - Y_in);
title('Y-axis Tracking Error [mm]');
linkaxes([ax1,ax2,ax3,ax4],'x')

