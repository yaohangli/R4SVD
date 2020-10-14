function [ U, S, V] = r4svd( X, tau, preU, percent)
% R4SVD for fast SVT operator estimation 
% Usage:  [U, S, V]  = r4svd(X, tau, preU, percent)
%
% Inputs:
%   X               - input sparse matrix
%   tau             - tau ||A||_* + 0.5 ||A||_F^2
%   preU            - recycle singular vector
%
% Outputs: singular value decomposition 
%   U               - n1xk left singular vectors 
%   S               - kx1 singular value vector
%   V               - n2xk right singular vectors 
%    
% SVT-R4SVD:
% Y. Li, W. Yu, A Fast Implementation of Singular Value Thresholding 
% Algorithm using Recycling Rank Revealing Randomized Singular Value 
% Decomposition, submitted to Mathematical Program Computation, 2017.

% Written by: Yaohang Li
% Email: yaohang@cs.odu.edu
% Created: Apr. 14, 2017

    np = 2;                             % number of power iterations
    [rows, cols] = size(X);             % rows and columns from X
    dk = int32(min(rows, cols)/20);     % 5% of dimension

    spreU = size(preU, 2);              % size of recycling vectors
    normX2 = norm(X, 'fro')^2;          % square of f-norm of X
      
    Omega = randn(cols,dk);             % random matrix
    Y = X*Omega;                        % project X onto omega
    for j = 1:np                        % power iterations
        Y = X*(X'*Y);
    end
    if spreU > 0                        % orthogonalization with recycled vectors
        Y = Y-preU*(preU'*Y);
    end
    
    [Q, R] = qr(Y, 0);                  % QR decomposition
    Q = [preU, Q];                      % build up approxmate basis
    B = Q'*X;                           % QB decomposition
    ratio = norm(B, 'fro')^2/normX2;    % error percentage
    k = spreU + dk;                     % initial rank
    
    % incrementally build up QB decomposition
    while (1 - ratio > percent)
        Omega = randn(cols,dk);         % random matrix
        Y = X*Omega;                    % projection
        for j = 1:np                    % power iterations
            Y = X*(X'*Y);
        end
        Y = Y - Q*(Q'*Y);               % orthogonalization with Q
        [QQ, R] = qr(Y, 0);             % QR decomposition               
        BB = QQ'*X;                     % QB decomposition
        ratio = ratio + norm(BB, 'fro')^2/normX2;   % error estimation
        Q = [Q, QQ];                    % build up Q
        B = vertcat(B, BB);             % build up B
        k = k + dk;                     % update rank
    end
   
    [Ub, Db, Vb] = svd(B, 'econ');      % SVD on short-and-wide
    protoU = Q*Ub;                      % approximate left singular vectors for X
    protoD = diag(Db);
    protoV = Vb;
        
    % check if there are any components with singular values greater
    % than the threshold
    idx = find(protoD >= tau,1,'last');
    
    if(isempty(idx))
        U = [];
        V = [];
        S = [];
        return;
    else
        V = protoV(:,1:idx);
        U = protoU(:,1:idx);
        S = protoD(1:1:idx);
    end
end

