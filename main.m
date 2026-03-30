clear;
clc;
close all;
% tic;


%% Adding Path
addpath(genpath('./datasets'));
addpath(genpath('./utils'));
% addpath(genpath('./SkewedT'));
% addpath(genpath('./proximal_operator'));

%% Load data
[imgname,pathname]=uigetfile('*.*', 'Select the  *.mat dataset','.\datasets');
str=strcat(pathname,imgname);
disp('The dataset is :')
disp(str);
addpath(pathname);
load(strcat(pathname,imgname));

%% Reading data
mask=double(map);
% mask=groundtruth;
[m,n,b]=size(data);
N=m*n;
dat=normalize(data);
Date = reshape(dat,N,b)';
%% 空间中心化（Spatial Centering）
% 计算每个波段的均值
mean_bands = mean(mean(dat, 1), 2);  % [1,1,b]
% 全局中心化：消除跨波段亮度变化
centered_cube = dat - mean_bands;     % [m,n,b]

%%空间平滑（3x3均值滤波）
smoothed_cube = zeros(size(centered_cube));
for bb = 1:b
    smoothed_cube(:,:,bb) = conv2(centered_cube(:,:,bb), ones(3)/9, 'same');
end

window_size = 7;  % 局部窗口大小 (2k+1, k=3)
epsilon = 1e-10;

% 计算每个波段的信息熵用于自适应alpha
entropy_per_band = zeros(b,1);
for bb = 1:b
    band_data = dat(:,:,bb);
    % 计算熵
    p = band_data(:) / sum(band_data(:));
    p = p(p > 0);  % 避免log(0)
    entropy_per_band(bb) = -sum(p .* log2(p + epsilon));
end

% 自适应alpha
alpha_adaptive = 1 + (entropy_per_band - min(entropy_per_band)) ./ (range(entropy_per_band) + epsilon);

% LCS增强
enhanced_cube = zeros(size(dat));
for bb = 1:b
    band_data = dat(:,:,bb);
    
    % 计算局部统计量 (2k+1窗口)
    local_mean = imfilter(band_data, fspecial('average', window_size), 'replicate');
    local_var = imfilter(band_data.^2, fspecial('average', window_size), 'replicate') - local_mean.^2;
    local_std = sqrt(max(local_var, 0)) + epsilon;
    
    % LCS公式: h_L = alpha * (h - mu) / (sigma + eps) + beta
    % beta设为0保持背景稳定
    enhanced_cube(:,:,bb) = alpha_adaptive(bb) * (band_data - local_mean) ./ local_std;
end

% 计算SNR
snr_pre = estimate_snr(smoothed_cube);
snr_lcs = estimate_snr(enhanced_cube);

% 融合权重
gamma = snr_pre / (snr_pre + snr_lcs);
Date_pre = reshape(smoothed_cube, N, b)';  % [b, N]
Date_lcs = reshape(enhanced_cube, N, b)';  % [b, N]

% 融合后的数据
Date_fused = gamma * Date_pre + (1 - gamma) * Date_lcs;
Date_fused = normalize(reshape(Date_fused', m, n, b));
Date1=Date_fused;
Date1=normalize(Date1);
Date2 = reshape(Date1,N,b)';


P = 3;

[M0, V, U, Y_bar, endm_proj, Y_proj] = find_endm(Date,P,'vca');
[M02, V2, U2, Y_bar2, endm_proj2, Y_proj2] = find_endm(Date2,P,'vca');
% [M02, V2, U2, Y_bar2, endm_proj2, Y_proj2] = find_endm(Date2,P,'nfindr');
% [M0, V, U, Y_bar, endm_proj, Y_proj] = find_endm(Date,P,'nfindr');
% [M02, V2, U2, Y_bar2, endm_proj2, Y_proj2] = find_endm(Date2,P,'nfindr');

M = M0;
M2 = M02;
% M=ones(b,2);

%%
% =========================================================================
% run the proposed method

L1=1;
L2=1e-5;
% learnRate = 1e-4;%0.1 0.01 0.001 0.0001
learnRate =1e-4;
% numEpochs= 100; %迭代次数1 50 100 200 300 400 500
numEpochs= 1; %迭代次数

lambda1=L1;    % OFF DIAG
lambda2=L1;    % OFF DIAG
lambda5=1e-2;  % inv(M)*M = I
lambda3=1e-2;  % W
lambda6=1e-6;  % SAD(M,M0)
rngflag=1;

disp('run MAC-U...') 

tic
[output,a_NNHU1,W,MLP1,rmse_r_NNHU1] = NNHU(Date,M,lambda1,lambda2,lambda3,lambda5,lambda6,learnRate,numEpochs,rngflag);
[output2,a_NNHU2,W2,MLP2,rmse_r_NNHU2] = NNHU(Date2,M2,lambda1,lambda2,lambda3,lambda5,lambda6,learnRate,numEpochs,rngflag);
time_NNHU1 = toc;

res = double(reshape(output,m,n,b));
% err2=(dat-res).^2;
err2=res.^2;
gae=mat2gray(sum(err2,3));
figure;imshow(gae);
figure; imagesc(gae);
axis image;

res2 = double(reshape(output2,m,n,b));
err22=res2.^2;
% gae2=RXfunc(err22);
gae2=mat2gray(sum(err22,3));

gae3=0.9*gae+0.1*gae2;   %SA 0.99 0.01

% toc;
det_map=reshape(gae3,N,1);
GT=reshape(mask,N,1);
mode_eq=1;
[AUC_D_F,AUC_D_tau,AUC_F_tau,AUC_TD,AUC_BS,AUC_SNPR,AUC_TDBS,AUC_ODP]=plot_3DROC(det_map,GT,mode_eq);

