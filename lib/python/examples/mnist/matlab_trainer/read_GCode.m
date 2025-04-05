function [values] = read_GCode(gcode)
%Function recieves a line of gcode and it extracts the relavent values.
%The output is an array. The output values are in the following order: G#, 
%F#, X# Y#, Z#, E#, and then I#, J#, K# (if IJK values exist). The function only 
%accepts G0, G1, G2, or G3 commands. The IJK format should be used for G2/G3 
%commands (not the R format).

%return if not G0 G1, G2, G3, G28, G90, G91 or G92 command
if ~strncmp(gcode, 'G0 ', 3) && ...
        ~strncmp(gcode, 'G1 ', 3) && ...
        ~strncmp(gcode, 'G2 ', 3) && ...
        ~strncmp(gcode, 'G3 ', 3) && ...
        ~strncmp(gcode, 'G28 ', 3) && ...
        ~strncmp(gcode, 'G90 ', 3) && ...
        ~strncmp(gcode, 'G91 ', 3) && ...
        ~strncmp(gcode, 'G92 ', 3)
    values(1) = NaN;
    return;
end

%get all the numbers from the string
input = sscanf(gcode, ...
            '%c %f %c %f %c %f %c %f %c %f %c %f %c %f %c %f %c %f');

%define the array 'values'  
% (1) G
% (2) F
% (3) X
% (4) Y
% (5) Z
% (6) E
% (7) I
% (8) J
% (9) K
values = zeros(1,9);
for i = 1:9
    values(i) = NaN;
end
values(1) = input(2); % G #

% for loop to put values in the correct output value
for i = 2:floor(length(input)/2);
    character = char(input(2*i - 1));
    if strcmp(character, 'F')
        values(2) = input(2*i);
    elseif strcmp(character, 'X')
        values(3) = input(2*i);
    elseif strcmp(character, 'Y')
        values(4) = input(2*i);
    elseif strcmp(character, 'Z')
        values(5) = input(2*i);
    elseif strcmp(character, 'E')
        values(6) = input(2*i);
    elseif strcmp(character, 'I')
        values(7) = input(2*i);
    elseif strcmp(character, 'J')
        values(8) = input(2*i);
    elseif strcmp(character, 'K')
        values(9) = input(2*i);
    end
end
    

       
end

