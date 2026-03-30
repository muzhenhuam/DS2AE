function local_entropy = calculateLocalEntropy(image, windowSize)
    % 将图像转换为双精度类型
    image = double(image);
    
    % 获取图像的尺寸
    [rows, cols] = size(image);
    
    % 初始化局部熵图像
    local_entropy = zeros(rows, cols);
    
    % 定义滑动窗口的半径
    halfWindowSize = floor(windowSize / 2);
    
    % 对图像每个像素点计算局部熵
    for i = 1:rows
        for j = 1:cols
            % 确定窗口的边界
            rmin = max(1, i - halfWindowSize);
            rmax = min(rows, i + halfWindowSize);
            cmin = max(1, j - halfWindowSize);
            cmax = min(cols, j + halfWindowSize);
            
            % 提取窗口中的局部区域
            window = image(rmin:rmax, cmin:cmax);
            
            % 计算局部区域的直方图
            histValues = imhist(window);
            histValues = histValues / numel(window); % 归一化
            
            % 计算局部区域的信息熵
            local_entropy(i, j) = -sum(histValues .* log2(histValues + eps)); % 加eps避免log(0)
        end
    end
end

% 示例使用
% image = imread('example_image.png'); % 读取图像
% if size(image, 3) == 3
%     image = rgb2gray(image); % 转换为灰度图像（如果是彩色图像）
% end
% 
% windowSize = 7; % 定义窗口大小
% local_entropy = calculateLocalEntropy(image, windowSize);
% 
% % 显示结果
% imshow(local_entropy, []);
% title('Local Entropy');
% colorbar;
