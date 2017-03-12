% Authors: Maunil Vyas
% Code: Implementation of Image reconstruction using technique by Bishop
% 
% Check by variying q size .................

%% setup
% clear; clc; close all;
OutputDir = '.\';

%Image file reading
T = (imread('PPCAInput.jpg'));

% convert to gray if it is not
if size(T,3) ~= 1
    fprintf('converting T to grayscale');
    T = rgb2gray(T);
    imwrite(T,'PPCAInput.jpg'); % save image
end
T = im2double(T);           % and convert uint8 to double

%% Define number of eigenvectors
eigen_vecs = 100;      % No of Basis (Eigen Vectors)
cd(OutputDir);               % where outputs will be saved

for J = 1:length(eigen_vecs)            % for all these lengths
    q = eigen_vecs(J);                  % set current length
                                        % and reconstruct the image
    mu = zeros(1, size(T,2));           % using # = 'q' eigen vectors
    
    for j = 1:size(T,2)
        mu(j) = mean(T(:,j));       %Computing the mean
    end
    
    S = zeros(size(T,2));
    
    % Compute Covariance matrix S
    for n = 1:size(T,1)
        S = S + (T(n,:)' - mu') * (T(n,:)' - mu')';
    end
    
    S = 1/size(T,1)*S;
    
	%% EM Algorithm to construct image
	No_iterations = 20;
	W_EM = randn(size(T,2),q);
	sigma_EM = randn(1);

	for i=1:1:No_iterations
		M_EM = W_EM'*W_EM + sigma_EM*eye(q);
		W_EM = S*W_EM/(sigma_EM*eye(q)+(M_EM\(W_EM'))*S*W_EM);
		sigma_EM = (1/size(S,1))*trace(S-S*W_EM/M_EM*W_EM');
	end
	
	% having obtained  M,W, sigma, we construct image
    Tnorm = zeros(size(T,1),size(T,2));
    
	for i = 1:size(T,1)
		 Tnorm(i,:) = T(i,:) - mu;
	end

	%Equation no 6 
	X = W_EM'*Tnorm';

	%Mentioned in page no 6 above section 4 last line
	rec = ((W_EM/(W_EM'*W_EM)*X))';

	for j = 1:size(T,1)
		rec(j,1:size(T,2)) = (rec(j,1:size(T,2))+mu(1:size(T,2)));
	end
	
	error = norm(rec-T);
	fprintf('EM Error: %f\n',error);
    imshow(rec);
	imwrite(rec,'PPCA_Output.jpg');
	
	%% Bishops method
    % Computing the eigen value and eigen vectors form the covariance matrix
    [e_ve,e_v] = eig(S);
    e_v = diag(e_v);
    
    [e_v, i] = sort(e_v, 'descend');
    e_ve = e_ve(:,i);
    
    U = e_ve(:,1:q);
    lambda_diag = diag(e_v);
    L = lambda_diag(1:q, 1:q);
    
    %two parameters by equation 7, equation 8 of paper
    sigma = sqrt(1/(size(S,1)-q)*sum(e_v(q+1:size(S,1))));
    W = U * sqrt(L - sigma^2*eye(q));
    
    %Value of M calculated equation given on page no 5 above section 3.2 last line
    M = W'*W + sigma^2 * eye(q);
    
    Tnorm = zeros(size(T,1),size(T,2));
    for i = 1:size(T,1)
        Tnorm(i,:) = T(i,:) - mu;
    end
    
    %Equation no 6
    X = (M\(W'))*(Tnorm');
    
    %Mentioned in page no 6 above section 4 last line
    rec = ((W/(W'*W)*M*X))';
    
    for j = 1:size(T,1)
        rec(j,1:size(T,2)) = (rec(j,1:size(T,2))+mu(1:size(T,2)));
    end
    % rec is the re-created image!
    
    %Computing the norm error
    Error = norm(rec-T);
    fprintf('Bishop Error: %f\n',Error);
    name = num2str(q);
    
    imwrite(rec,sprintf('%s.JPG',name));
end

cd '..'