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

% [H,W,Dim]=size(DataTest);
% num=H*W;
% windowSize = 3; % 定义窗口大小
% for i=1:b
%     DataTest(:,:,i) = calculateLocalEntropy(dat(:,:,i), windowSize);
% end
% Date = reshape(DataTest,N,b)';
Date = reshape(dat,N,b)';
% Extract EMs from each class

%%1
% 预处理
Date2 = pre(Date);


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
[output,a_NNHU1,W,MLP1,rmse_r_NNHU1] = NNHU_autoencoder_customize_1(Date,M,lambda1,lambda2,lambda3,lambda5,lambda6,learnRate,numEpochs,rngflag);
[output2,a_NNHU2,W2,MLP2,rmse_r_NNHU2] = NNHU_autoencoder_customize_1(Date2,M2,lambda1,lambda2,lambda3,lambda5,lambda6,learnRate,numEpochs,rngflag);
time_NNHU1 = toc;

res = double(reshape(output,m,n,b));
% figure;imshow(res(:,:,100),[]);
% figure;imshow(data(:,:,100),[]);
% figure;imshow(mask,[]);
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

gae3=fusion(gae,gae2);  

% toc;
det_map=reshape(gae3,N,1);
GT=reshape(mask,N,1);
mode_eq=1;
[AUC_D_F,AUC_D_tau,AUC_F_tau,AUC_TD,AUC_BS,AUC_SNPR,AUC_TDBS,AUC_ODP]=plot_3DROC(det_map,GT,mode_eq);
%%
RR=gae3;
figure
imagesc(RR)
[r,c] = size(RR);
axis image %这里是使图像以原比例显示
set(gca,'XTick',[],'YTick',[])%去掉横纵坐标的刻度线
set(gca,'Position',[0 0 1 1])%让图像充满整个图窗
set(gcf,'Position',[300 300 400 400]);%消除白边
set(gcf,'innerposition',[100,500,c,r])%让图像充满整个图窗
EUF1=RR;
saveas(gca,'EUF.bmp')%保存图像
save EUF EUF1