function [e_rms, T_print] = Simulate_Output_CISCO(input_filename, input_printer, feedrate, acceleration)
% clear all
close all
clc

%% Load trajecotry
filename = input_filename; %'xyzCalibration_cube';
F_max = feedrate; % 150; % maximum feedrate [mm/s]
A_max = acceleration; % 10000;    % maximum x and y acceleration [mm/s^2]
load(['Discrete_Position_',filename,'_V_',num2str(F_max),'_A_',num2str(A_max),'.mat'])

%% Load simulated system dynamics for the investigated printer
zeta = 0.3;

% 10-15 pretty bad; 15-20 somehow bad; 20-30 medium; 
printer_set = [15, 15; 
    15, 25; 
    25, 20; 
    50, 50; 
    40, 50; 
    55, 45; 
    25, 25; 
    27, 22; 
    15, 30; 
    60, 50];

% Select specific printer number
printer_no = input_printer;

% Extract frequencies for the selected printer
f_x = printer_set(printer_no, 1);
f_y = printer_set(printer_no, 2);
fprintf("Printer No. %d: f_x = %.1f Hz, f_y = %.1f Hz.\n", printer_no, f_x, f_y);

omega_x = 2 * pi * f_x;
omega_y = 2 * pi * f_y;

num_x = omega_x^2;
den_x = [1, 2*zeta*omega_x, omega_x^2];
sys_x1 = tf(num_x, den_x);
num_y = omega_y^2;
den_y = [1, 2*zeta*omega_y, omega_y^2];
sys_y1 = tf(num_y, den_y);

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
%writematrix(Data,['IO_Data_Printer_',num2str(printer_no),'_',filename,'.csv']);

% Define the sampling frequency and the cutoff frequency
Fs = 1/(Time(2)-Time(1));  % Example sampling frequency in Hz
Fc = max(printer_set(:))*1.5;              % Cutoff frequency in Hz

% Design a low-pass filter
d = designfilt('lowpassiir', 'FilterOrder', 6, ...
                'HalfPowerFrequency', Fc, ...
                'SampleRate', Fs);
X_out_lp = filtfilt(d,X_out);
Y_out_lp = filtfilt(d,Y_out);

%% Calculate for the contour error of the simulated trajectory
start_idx = find(Z_in == z_e_start, 1, 'first');
end_idx = find(Z_in == z_e_end, 1, 'last');
theta_in = zeros(length(X_in),1);
for i = 2:length(X_out_lp)
    theta_in(i) = atan2((Y_in(i) - Y_in(i-1)),(X_in(i) - X_in(i-1))); % angle between trajectories represented by blocks (between -pi and pi)
    if theta_in(i) < 0
        theta_in(i) = 2*pi + theta_in(i); % convert angle to be between 0 and 2pi
    end
end
e_x = X_in(start_idx:end_idx)-X_out_lp(start_idx:end_idx);
e_y = Y_in(start_idx:end_idx)-Y_out_lp(start_idx:end_idx);
e_rms = rms(-sin(theta_in(start_idx:end_idx)).*e_x+cos(theta_in(start_idx:end_idx)).*e_y);
T_print = Time(end_idx) - Time(start_idx);
fprintf("Total time: %.1f (s); RMS of the contour error: %.3g (mm).\n", T_print, e_rms);
%% Plot for visualization
% figure;
% subplot(121); 
% plot3(X_in,  Y_in,  Z_in ); 
% title('Nominal shape'); axis equal;
% subplot(122); 
% plot3(X_out, Y_out, Z_out);
% title('Output shape'); axis equal;

% figure;
% subplot(221); ax1 = gca;
% plot(Time, X_in); hold on;
% plot(Time, X_out);
% legend('Input','Output');
% title('X-axis Input/Output Trajectory [mm]');
% subplot(223); ax3 = gca;
% plot(Time, X_out - X_in); hold on;
% plot(Time, X_out_lp - X_in); hold off;
% title('X-axis Tracking Error [mm]');
% subplot(222); ax2 = gca;
% plot(Time, Y_in); hold on;
% plot(Time, Y_out);
% legend('Input','Output');
% title('Y-axis Input/Output Trajectory [mm]');
% subplot(224); ax4 = gca;
% plot(Time, Y_out - Y_in); hold on;
% plot(Time, Y_out_lp - Y_in); hold off;
% title('Y-axis Tracking Error [mm]');
% linkaxes([ax1,ax2,ax3,ax4],'x')
end