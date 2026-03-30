function enhanced_band = LCS_single_band(band, window_size, alpha, beta)
% 局部对比度拉伸（单波段）
% 输入:
%   band          - 输入单波段图像矩阵 (H x W)，含NaN需处理
%   window_size   - 滑动窗口大小（奇数，如5）
%   alpha         - 对比度缩放因子（默认1.0）
%   beta          - 偏移量（若为空则用局部均值）
% 输出:
%   enhanced_band - 增强后的单波段图像

if nargin < 3
    alpha = 1.0;
end

% 计算局部均值和标准差（手动实现NaN忽略）
half_win = floor(window_size/2);
[H, W] = size(band);
local_mean = zeros(H, W);
local_std = zeros(H, W);

for i = 1:H
    for j = 1:W
        % 获取当前窗口
        row_start = max(1, i-half_win);
        row_end = min(H, i+half_win);
        col_start = max(1, j-half_win);
        col_end = min(W, j+half_win);
        
        window = band(row_start:row_end, col_start:col_end);
        window = window(~isnan(window)); % 忽略NaN值
        
        % 计算统计量
        if ~isempty(window)
            local_mean(i,j) = mean(window(:));
            local_std(i,j) = std(window(:), 0);
        else
            local_mean(i,j) = NaN;
            local_std(i,j) = NaN;
        end
    end
end

% 设置默认偏移量（局部均值）
if nargin < 4 || isempty(beta)
    beta = local_mean;
end

% LCS变换
epsilon = 1e-6;
enhanced_band = alpha .* (band - local_mean) ./ (local_std + epsilon) + beta;

% 限制输出范围并保留NaN
enhanced_band(isnan(band)) = NaN;
enhanced_band = max(min(enhanced_band, 1), 0); % 假设输入已归一化到[0,1]
end