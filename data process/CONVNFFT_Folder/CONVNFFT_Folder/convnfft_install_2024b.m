function convnfft_install
% function convnfft_install
% Installation by building the C-mex file needed for convnfft
%
% Author: Bruno Luong <brunoluong@yahoo.com>
% History
%  Original: 16/Sept/2009

% Step 1: Determine system architecture
arch = computer('arch');  % Get system architecture (e.g., 'glnxa64', 'win64')

% Step 2: Set compilation options
mexopts = {'-O', '-v', ['-' arch]};  % Optimization flag and architecture-specific options

% Step 3: Check MATLAB version
if ~verLessThan('MATLAB', '9.4')  % If MATLAB version >= R2024b
    R2024b_mexopts = {};  % No need for R2024b specific options
else
    % Step 4: Check for 64-bit platform
    if ~isempty(strfind(computer(), '64'))  % If 64-bit platform
        mexopts(end+1) = {'-largeArrayDims'};  % Option for large array support
    end
    R2024b_mexopts = {};  % No additional options for older versions
end

% Step 5: Compile the C-mex file
mex(mexopts{:}, 'inplaceprod.c');  % Compile inplaceprod.c file with options
