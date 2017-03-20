function [y,mu,W,S,sigma,x_t] = fn_PPCA_EM_cor(Im,q,No_iteration)
    
    %Computing the mean value
    row = size(Im,1);
    col = size(Im,2);         
    mu = zeros(row,1);      % Mean mu
    
    W = randn(row,q);           % Initializing W 
    sigma = randn(1);          % Initializing Sigma (Variance)
    
    for i = 1:1:col
        for j = 1:1:row
            if (Im(j,i)>=0)            
                mu(j,1) = mu(j,1) + Im(j,i);
            end
        end
    end
    
    mu = mu/col;
    
    %Computing the Covariance matrix as mentioned in the first para of the
    %paper 
    
    S = zeros(row,row);
    %col_vec = zeros(row,1);
    
    for i=1:1:col
        col_vec = Im(:,i);
        for j=1:1:row
            if (col_vec(j,1)<0)
                col_vec(j,1) = mu(j,1);
            end
        end
        S = S + (col_vec-mu)*(col_vec-mu)';
    end
    S = S/col;
    
    for i = 1:1:No_iteration
        M = W'*W + sigma*eye(q);
        W = S*W/(sigma*eye(q)+M\(W')*S*W);
        sigma = (1/size(S,1))*trace(S-S*W/M*W');
    end
    
    Tnorm = zeros(row,col);
    
    for i = 1:1:col
        for k = 1:1:row
            if (Im(k,i)>=0)        
                Tnorm(k,i) = (Im(k,i)- mu(k));
            end
        end
    end
    
    
    %Equation no 6 
    x_t = W'*Tnorm;
    
    %Reconstruction Mentioned in page no 6 above section 4 last line
    y = ((W/(W'*W))*x_t);

    for j = 1:size(Im,2)
        y(:,j) = y(:,j)+mu;
    end
    
end