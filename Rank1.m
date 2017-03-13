function [Un,Sn,Vn] = Rank1(U, S, V, a, b)

    VaD = [V' zeros(size(V',1),1)];
    V = VaD';  
% From Equation 6
    m = U'*a;
    p = a - U*m;
    Ra = norm(p);
    P = inv(Ra)*p;
% From Equation 7
    n = V'*b;
    q = b - V*n;
    Rb = norm(q);
    Q = inv(Rb)*q;

    K=[S zeros(size(S,1),1);zeros(1,size(S,2)+1)]+[m;Ra]*([n;Rb]'); % Equation 8 
    [UDn, SDn, VDn] = svd(K);

    Up = [U P];
    Vq = [V Q];

    Un = Up*UDn;    
    Vn = Vq*VDn;    
    Sn = SDn;       

end