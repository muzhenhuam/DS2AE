function snr_val = estimate_snr(cube)
%ESTIMATE_SNR 估计高光谱图像的信噪比
%   输入: cube - 高光谱数据立方体 [m, n, b]
%   输出: snr_val - 估计的SNR值（标量）
%
%   基于空间梯度法估计噪声功率，信号功率使用方差估计

    [m, n, b] = size(cube);
    

    noise_power = 0;
    for bb = 1:b
        band = cube(:,:,bb);
    
        dx = diff(band, 1, 2);     
        dy = diff(band, 1, 1);     
        
       
        noise_power_bb = (sum(dx(:).^2) + sum(dy(:).^2)) / (numel(dx) + numel(dy));
        noise_power = noise_power + noise_power_bb;
    end
    noise_power = noise_power / b; 
    
 
    signal_power = var(cube(:));
    
    snr_val = signal_power / (noise_power + 1e-10);
    
 
    % % 使用3x3局部窗口分离信号和噪声
    % signal_power2 = 0;
    % noise_power2 = 0;
    % for bb = 1:b
    %     band = cube(:,:,bb);
    %     % 局部均值（信号）
    %     local_mean = imfilter(band, fspecial('average', 3), 'replicate');
    %     % 残差（噪声）
    %     residual = band - local_mean;
    %     
    %     signal_power2 = signal_power2 + var(local_mean(:));
    %     noise_power2 = noise_power2 + var(residual(:));
    % end
    % snr_val2 = (signal_power2/b) / (noise_power2/b + 1e-10);
end