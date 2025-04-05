% Purpose: Read .gcode file (or .txt file containing GCode), convert it
% into XYZE coordinates and feed rate, and then store them in .mat file.
% All the units are metric. Feedrate = mm/min, coordinates are in mm

close all
clear all
clc

tic

%% Load gcode file
filename = 'xyzCalibration_cube';

%read in gcode from text file
gcode = textread([filename,'.gcode'],'%s','delimiter','\n');

%% Read gcode line by line
numCommands = length(gcode);
command = zeros(1,9);
% (1) G
% (2) F
% (3) X
% (4) Y
% (5) Z
% (6) E
% (7) I
% (8) J
% (9) K

k = 1;
for i = 1:numCommands
    % Convert string into array of command values
    temp_command = read_GCode(char(gcode(i))); % output is [G,F,X,Y,Z,E,I,J,K]
    if ~isnan(temp_command(1)) % discard command blocks that are not related to axis movement
        command = vertcat(command, temp_command);
        k = k + 1; % increment counter
    end

end
numCommands = k; % real number of commands

%% Extract key points from gcode
% replace coordinates or feedrate with the previous row's numbers if
% absolute positinoing mode
x_curr = 0; % initialize x
y_curr = 0; % initialize y
z_curr = 0; % initialize z
e_curr = 0; % initialize e
f_curr = 0; % initialize f
mode = 1; % (1) = absolute positioning, (0) = relative positioning
for i = 1:numCommands-1
    if command(i+1,1) == 90
        mode = 1;
    elseif command(i+1,1) == 91
        mode = 0;
    elseif command(i+1,1) == 28
        mode = 1;
    end
    
    if mode > 0.5 
        if isnan(command(i+1,2))
            command(i+1,2) = f_curr;
        end
        if isnan(command(i+1,3))
            command(i+1,3) = x_curr;
        end
        if isnan(command(i+1,4))
            command(i+1,4) = y_curr;
        end
        if isnan(command(i+1,5))
            command(i+1,5) = z_curr;
        end
        if isnan(command(i+1,6))
            command(i+1,6) = e_curr;
        end
        if isnan(command(i+1,7))
            command(i+1,7) = command(i,7);
        end
        if isnan(command(i+1,8))
            command(i+1,8) = command(i,8);
        end
        if isnan(command(i+1,9))
            command(i+1,9) = command(i,9);
        end
        x_curr = command(i+1,3); % update x
        y_curr = command(i+1,4); % update y
        z_curr = command(i+1,5); % update z
        e_curr = command(i+1,6); % update e
        f_curr = command(i+1,2); % update f
        
    else
        command(i+1,1) = 90;
        if isnan(command(i+1,2))
            command(i+1,2) = f_curr;
        end
        if isnan(command(i+1,3))
            command(i+1,3) = x_curr;
        else
            x_curr = x_curr + command(i+1,3); % update x
            command(i+1,3) = x_curr;
        end
        if isnan(command(i+1,4))
            command(i+1,4) = y_curr;
        else
            y_curr = y_curr + command(i+1,4); % update y
            command(i+1,4) = y_curr;
        end
        if isnan(command(i+1,5))
            command(i+1,5) = z_curr;
        else
            z_curr = z_curr + command(i+1,5); % update z
            command(i+1,5) = z_curr;
        end
        if isnan(command(i+1,6))
            command(i+1,6) = e_curr;
        else
            e_curr = e_curr + command(i+1,6); % update e
            command(i+1,6) = e_curr;
        end
        if isnan(command(i+1,7))
            command(i+1,7) = command(i,7);
        end
        if isnan(command(i+1,8))
            command(i+1,8) = command(i,8);
        end
        if isnan(command(i+1,9))
            command(i+1,9) = command(i,9);
        end
        
    end
    
end
toc

G = command(:,1);
F = command(:,2);
X = command(:,3);
Y = command(:,4);
Z = command(:,5);
E = command(:,6);

save(['gcode_data_',filename,'.mat'],'G','F','X','Y','Z','E')

fileID = fopen(['gcode_data_',filename,'.txt'], 'w');

fprintf(fileID, 'G F X Y Z E\n');
for i = 1:length(G)
    fprintf(fileID, '%d %d %f %f %f %f\n', G(i,1), F(i,1), X(i,1), Y(i,1), Z(i,1), E(i,1));
end

fclose(fileID);
%% Plot for visualization
figure;
plot3(X,Y,Z); 
axis equal;

figure;
subplot(411); plot(X); ax1 = gca;
subplot(412); plot(Y); ax2 = gca;
subplot(413); plot(Z); ax3 = gca;
subplot(414); plot(F); ax4 = gca;
linkaxes([ax1,ax2,ax3,ax4],'x');



