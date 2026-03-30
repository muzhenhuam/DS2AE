function enhanced_cube = LCS_hyperspectral(hsi_cube, window_size, alpha)
% 高光谱数据局部对比度拉伸（全波段）
% 输入:
%   hsi_cube     - 高光谱数据立方体 (H x W x B)，可含NaN
%   window_size  - 滑动窗口大小（奇数）
%   alpha        - 统一缩放因子或波段自适应向量（B x 1）
% 输出:
%   enhanced_cube - 增强后的高光谱数据（NaN保留）

[H, W, B] = size(hsi_cube);
enhanced_cube = zeros(H, W, B);

% 检查alpha是否为标量
if isscalar(alpha)
    alpha = repmat(alpha, B, 1);
end

% 并行逐波段处理
parfor b = 1:B
    enhanced_cube(:,:,b) = LCS_single_band(hsi_cube(:,:,b), window_size, alpha(b));
end
end