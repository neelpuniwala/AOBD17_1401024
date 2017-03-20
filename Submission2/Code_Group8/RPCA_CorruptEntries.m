% Script to test RPCA for corrupt entries in input image.
clear; clc; close all;

Err_Rate = 10;			% PERCENTAGE of error rate

Im = im2double(imread('cameraman.tif'));
img_miss = Im;

% generate random indexes and replace them with NULL
rng('default'); 										% for reproducibility
ix = random('unif',0,1,size(img_miss))<(Err_Rate/100);
img_miss(ix) = randn(1);

%% create a matrix X from overlapping patches
% Segmentation
ws = 16;					%window size = 16 * 16

% find the number of patches
no_patches = (size(img_miss, 1) / ws);

k = 1;		% index for allocation
X = zeros(no_patches^2, ws^2);

for i = (1:no_patches*2-1)
    for j = (1:no_patches*2-1)
        r1 = 1+(i-1)*ws/2:(i+1)*ws/2;
        r2 = 1+(j-1)*ws/2:(j+1)*ws/2;
        patch = img_miss(r1, r2);
        X(k,:) = patch(:);
        k = k + 1;
    end
end

%% RPCA
lambda = 0.0625; 				% lambda = 1/sqrt(max(dim)) % dim is constant 256*256 or something
tic
[L, S] = fn_RobustPCA(X, lambda, 1.0, 1e-5);
toc

% reconstruct the image from the overlapping patches in matrix L
img_remade = zeros(size(Im));
k = 1;

for i = (1:no_patches*2-1)
    for j = (1:no_patches*2-1)
        % average patches to get the image back from L and S
        % todo: in the borders less than 4 patches are averaged
        patch = reshape(L(k,:), ws, ws);
        r1 = 1+(i-1)*ws/2:(i+1)*ws/2;
        r2 = 1+(j-1)*ws/2:(j+1)*ws/2;
        img_remade(r1, r2) = img_remade(r1, r2) + 0.25*patch;
        patch = reshape(S(k,:), ws, ws);
        k = k + 1;
    end
end

%% find error
error = Im - img_remade;
RMS_error = sqrt(sum(sum(error.^2))/(size(Im,1)*size(Im,2)));
disp('error');
disp(RMS_error);

% show the results
figure; imshow(img_miss), title('Corrupted image')
figure; imshow(img_remade), title('Recovered image')
