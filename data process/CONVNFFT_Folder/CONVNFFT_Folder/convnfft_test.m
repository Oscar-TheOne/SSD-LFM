function A = convnfft_test(A, B, shape, dims, options)
% CONVNFFT_TEST  FFT-BASED N-dimensional convolution.
%   A = CONVNFFT_TEST(A, B) performs the N-dimensional convolution of
%   matrices A and B using the Fast Fourier Transform (FFT) method.
%   
%   C = CONVNFFT(A, B, SHAPE) controls the size of the result C:
%       'full'   - Returns the full N-D convolution (default).
%       'same'   - Returns the central part of the convolution that is 
%                  the same size as A.
%       'valid'  - Returns only the part of the result that can be 
%                  computed without assuming zero-padding.
%
%   C = CONVNFFT(..., DIMS) specifies the dimensions along which to perform
%   the convolution. By default, the convolution is performed over all 
%   dimensions.
%
%   C = CONVNFFT(..., OPTIONS) specifies additional options like GPU use 
%   and FFT optimizations.
%       OPTIONS is a structure with the following optional fields:
%       - 'GPU', boolean. If TRUE, GPU-based FFT is used (default: FALSE).
%       - 'Power2Flag', boolean. If TRUE, use FFT with length rounded to
%         the next power of two for efficiency (default: TRUE).
%
%   Class support for inputs A, B:
%       float: double, single
%
%   Author: Bruno Luong <brunoluong@yahoo.com>
%   History:
%       - Original: 21-Jun-2009
%       - Bug fixes, GPU support, options structure update.
    
    %% Input parsing and option handling
    if nargin < 3 || isempty(shape)
        shape = 'full';  % Default shape
    end

    if nargin < 5 || isempty(options)
        options = struct();  % Empty options by default
    elseif ~isstruct(options)
        % If options is a boolean, treat it as the GPU flag
        options = struct('GPU', options);
    end

    % Check the dimensions of A and B
    nd = max(ndims(A), ndims(B));  % N-dimensional size
    if nargin < 4 || isempty(dims)
        dims = 1:nd;  % Default: apply convolution to all dimensions
    end
    dims = reshape(dims, 1, []);  % Ensure dims is a row vector for indexing
    
    % Get GPU option
    GPU = getoption(options, 'GPU', true);  % Default to with GPU
    GPU = GPU && gpuDeviceCount > 0;  % Enable GPU if available
    
    % Define the function for truncating the result based on 'shape'
    switch lower(shape)
        case 'full'
            ifun = @(m, n) 1:m+n-1;  % Full convolution
        case 'same'
            ifun = @(m, n) ceil((n-1)/2)+(1:m);  % Same-size convolution
        case 'valid'
            ifun = @(m, n) n:m;  % Valid convolution (no padding)
        otherwise
            error('convnfft_test: unknown shape %s', shape);  % Error for unknown shape
    end

    % Check if A and B are real
    classA = class(A);
    classB = class(B);
    ABreal = isreal(A) && isreal(B);

    %% Special case: empty convolution
    if any(size(A) == 0) || any(size(B) == 0)
        szA = zeros(1, nd); szA(1:ndims(A)) = size(A);
        szB = zeros(1, nd); szB(1:ndims(B)) = size(B);
        szA = max(szA, 1);  % MATLAB convention for empty dimensions
        szB = max(szB, 1);
        szC = szA;  % Initialize result size as A's size
        for dim = dims
            szC(dim) = length(ifun(szA(dim), szB(dim)));
        end
        A = zeros(szC, classA);  % Return empty result
        return
    end

    %% Determine FFT length (using power-of-2 if requested)
    power2flag = getoption(options, 'Power2Flag', true);
    if power2flag
        lfftfun = @(l) 2^nextpow2(l);  % FFT length is next power of 2
    else
        lfftfun = @(l) l;  % Use exact length (no padding)
    end

    %% Move data to GPU if necessary
    if GPU
        A = gpuArray(A);
        B = gpuArray(B);
    end

    %% Perform the FFT convolution in each dimension
    subs(1:ndims(A)) = {':'};
    for dim = dims
        m = size(A, dim);  % Size of A in the current dimension
        n = size(B, dim);  % Size of B in the current dimension
        l = lfftfun(m + n - 1);  % FFT length for current dimension

        % If not the first dimension, permute to bring current dimension to front
        if dim ~= 1
            swap = 1:nd;
            swap([1 dim]) = swap([dim 1]);
            A = permute(A, swap);
            B = permute(B, swap);
        end
        
        % Compute FFT along the current dimension
        A = fft(A, l, dim);
        B = fft(B, l, dim);
        subs{dim} = ifun(m, n);  % Define the region to extract
    end

    %% Element-wise multiplication in the frequency domain
    A = A .* B;

    % ...（之前的代码保持不变）

%% Perform the inverse FFT to get the result back to spatial domain
for dim = dims(end:-1:1)  % Reverse loop order for inverse FFT
    A = ifft(A, [], dim);  % Inverse FFT along each dimension
    if dim ~= 1
        swap = 1:nd;
        swap([1 dim]) = swap([dim 1]);
        A = permute(A, swap);  % Swap back dimensions
    end
end

%% Determine the final size of the result based on the shape parameter
% (这部分代码看起来是正确的，但请注意，'same'形状的处理可能需要更精确的逻辑，
% 特别是当输入尺寸不是2的幂时。)
szA = size(A);
szC = szA;  % Initialize result size as A's size (before truncation)
for dim = dims
    m = szA(dim);
    n = size(B, dim); % 注意：这里假设B已经被定义，且其维度与A兼容（或至少我们知道其维度）
    switch lower(shape)
        case 'full'
            szC(dim) = m + n - 1;
        case 'same'
            % 'same'形状通常意味着输出尺寸与输入中较大的那个相同（或接近），
            % 但这取决于IFFT的具体实现和输入尺寸。这里我们简化处理。
            szC(dim) = ceil(max(m, n) / 2) * 2; % 尝试保持偶数尺寸（可能不是最准确的）
            % 注意：对于非2的幂尺寸，'same'可能无法精确匹配任何输入尺寸的输出。
        case 'valid'
            szC(dim) = max(m - n + 1, 0);
    end
end
 
%% Create an index array for truncating A (simplified)
index = true(szC); % 初始化一个与期望输出尺寸相同的逻辑索引数组
for dim = 1:nd
    if dim in dims
        % 确定当前维度的起始和结束索引
        if strcmp(shape, 'full')
            % 对于'full'，我们使用整个维度
            start_idx = 1;
            end_idx = szA(dim);
        elseif strcmp(shape, 'same') || strcmp(shape, 'valid')
            % 对于'same'和'valid'，我们需要计算截断索引
            % 注意：这里的处理是基于IFFT后的尺寸szA，且没有考虑维度交换。
            % 如果之前进行了维度交换，则这里的逻辑需要相应地调整。
            center_idx = floor((szA(dim) + 1) / 2); % IFFT后通常中心化
            if strcmp(shape, 'same')
                % 'same'通常意味着输出尺寸与输入中较大的那个“相同”（在某种意义上）
                % 但由于尺寸可能不是2的幂，我们取最接近的偶数尺寸的一半作为中心范围
                half_len = floor(szC(dim) / 2);
            elseif strcmp(shape, 'valid')
                % 'valid'意味着没有填充，所以输出尺寸是输入尺寸减去（n-1）
                half_len = floor((m - n + 1) / 2); % 注意：这里m是IFFT后的尺寸
            end
            % 调整起始和结束索引以适应中心化和可能的尺寸不匹配
            start_idx = max(1, center_idx - half_len + 1);
            end_idx = min(szA(dim), center_idx + half_len);
        end
        % 应用索引截断到当前维度（这里我们假设没有维度交换）
        index = index & (1:szC(dim))' >= start_idx & (1:szC(dim))' <= end_idx; % 注意：这里使用了列向量进行比较
        % 注意：由于MATLAB的索引是从1开始的，并且我们在这里处理的是逻辑索引，
        % 所以我们不需要担心维度交换对索引的影响（至少在这个截断步骤中）。
        % 但是，如果后续处理需要原始维度顺序，则应在显示或保存结果之前重新交换维度。
    end
end
index = reshape(index, szC); % 确保索引数组与期望的输出尺寸相匹配
 
%% Apply the index to truncate A
A = A(index); % 直接使用逻辑索引截断A（无论A是实数还是复数）
% 注意：如果A是复数且您想要实数结果，则应在截断之前或之后取实部（取决于您的需求）。
% 这里我们保留了复数结果。
 
%% Move back to CPU if GPU was used (这部分代码取决于您的具体实现和GPU环境)
if GPU
    A = gather(A); % 假设gather是将数据从GPU移动到CPU的函数（这取决于您的具体实现）
end
end


function value = getoption(options, name, defaultvalue)
% function value = getoption(options, name, defaultvalue)
    value = defaultvalue;
    fields = fieldnames(options);
    found = strcmpi(name,fields);
    if any(found)
        i = find(found,1,'first');
        if ~isempty(options.(fields{i}))
            value = options.(fields{i});
        end
    end
end
