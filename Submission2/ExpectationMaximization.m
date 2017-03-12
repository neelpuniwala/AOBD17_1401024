% Author: Raj Derasari 

%% setup
clc; clear;close all;
addpath('EM_Functions');

%% load components for GMM (gaussian mixture model)
load GSModel_8x8_200_2M_noDC_zeromean.mat
GMM.ncomponents = GS.nmodels;
GMM.mus = GS.means;
GMM.covs = GS.covs;
GMM.weights = GS.mixweights;
clear GS; % remove struct from memory

%% take input image
% YOU CAN ALSO use any of the images provided in 
CURPATH = pwd();

% generate a random, appropriate index (should be > 2)
% imgpath = '.\Image_Dataset\'; % in order to use img repository, end with \
% Flist = dir(imgpath);
% Ind = randi([3 length(Flist)],1,1);
% filename = Flist(Ind).name;
% fprintf('input image from repo: %s',filename);
% imgpath = strcat(CURPATH,char(imgpath),filename);


% % USE THIS, when image is defined (apna Lena256.png)
imgpath = '\';          % dont use '.', only '\' for current directory
filename = 'EMInput.jpg'; % DO include file type with name
imgpath = strcat(CURPATH,char(imgpath),filename);
fprintf('input image: %s',imgpath);

%% Load image and proceed..
x = imread(imgpath);
if size(x,3) ~= 1
    x = rgb2gray(x);
end
x = im2double(x);
clearvars imgpath;

%% image processing begins now

% fill input image with noise
sigmaNoise = 20/255;            % noise variance
Sigmas = [1, 0.5, 0.3535534, 0.25, 0.1767767]; 			% (1, 1/2, 1/2r2, 1/4, 1/4r2
y = x + sigmaNoise * randn(size(x));        % noisy test image

%% EPLL ; Expected Patch Log Likelihood // denoising algorithm, works on GMM
xEPLL = y;
for sigma = sigmaNoise * [1, 1/sqrt(4), 1/sqrt(8), 1/sqrt(16), 1/sqrt(32)]
    xEPLL = MAP_GMM(x, y, xEPLL, sigmaNoise, sigma, GMM);
end
% XEPLL is the obtained/output image of denoising algo

%% EM adaptation using EPLL output + MAP-denoising with adapted GMM
xHat = xEPLL;
epsilon = 0.01;
b = randn(size(y));
n = numel(y);
xEPLL1 = y + epsilon*b;
for sigma = sigmaNoise * [1, 1/sqrt(4), 1/sqrt(8), 1/sqrt(16), 1/sqrt(32)]
    xEPLL1 = MAP_GMM(x, y + epsilon*b, xEPLL1, sigmaNoise, sigma, GMM);
end
xHat1 = xEPLL1;
div = (b(:)'*(xHat1(:) - xHat(:))) / (n*epsilon);
beta_opt = (sqrt(mean((y(:) - xHat(:)).^2) - sigmaNoise^2 + 2*sigmaNoise^2*div)) / sigmaNoise;
aGMM = EM_adaptation(GMM, xEPLL, beta_opt * sigmaNoise, 1);
xAdapted_EPLL = y;
for sigma = sigmaNoise * [1, 1/sqrt(4), 1/sqrt(8), 1/sqrt(16), 1/sqrt(32)]
    xAdapted_EPLL = MAP_GMM(x, y, xAdapted_EPLL, sigmaNoise, sigma, aGMM);
end
%clearvars Sigmas sigma sigmaNoise n epsilon;
imwrite(xEPLL,'X_EPLL_Zoran-Yeis.jpg');
imwrite(xAdapted_EPLL,'X_EPLL_EM_E-Lao.jpg');

figure,
subplot(2,2,1); imshow(x); title('Input Image');
subplot(2,2,2); imshow(y); title('Input Made Very Noisy!');
subplot(2,2,3); imshow(xEPLL); title('Obtained by EPLL');
subplot(2,2,4); imshow(xAdapted_EPLL); title('Obtained by EPLL + EM algorithm');

clearvars b  XEPLL XEPLL1 xHat y
return;
