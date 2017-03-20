% Script to test EM for missing entries in input image.
clear; clc; close all;

Err_Rate = 10;			% PERCENTAGE of error rate

Im = im2double(imread('cameraman.tif'));
img_miss = Im;

% generate random indexes and replace them with NULL
rng('default'); 										% for reproducibility
ix = random('unif',0,1,size(img_miss))<(Err_Rate/100);
img_miss(ix) =NaN;

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

% EM algorithm 
q = 200;				% desired rank of output matrix
%q = sqrt(1/2) * img_miss;
Max_iterations = 50;
% use tic and toc to find time difference
tic
[y,mu,W,S,sigma,x_t] = fn_PPCA_EM_missing(X,q,Max_iterations);
toc

%% Reconstruction of image
img_remade = zeros(size(img_miss));
k = 1;
for i = (1:no_patches*2-1)
	for j = (1:no_patches*2-1)
		patch = reshape(y(k,:), ws, ws);
		r1 = 1+(i-1)*ws/2:(i+1)*ws/2;
		r2 = 1+(j-1)*ws/2:(j+1)*ws/2;
		img_remade(r1, r2) = img_remade(r1, r2) + 0.25*patch;
		k = k + 1;
	end
end

%% find the error
error = Im - img_remade;
RMS_error = sqrt(sum(sum(error.^2))/(size(img_miss,1)*size(img_miss,2)));
disp('error:');
disp(RMS_error);

%% show the results
figure;imshow(img_miss), title('Corrupted image');
figure;imshow(img_remade), title('Recovered image');
