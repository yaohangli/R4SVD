function [Ubest,Sbest,Vbest,numiter,out] = SVT_r4svd(n,trIndex,trData,tsIndex,tsData,tau,delta,maxiter,tol)
%% SVT Implementation using R4SVD for fast SVT operator estimation 
% Usage:  [Ubest,Sbest,Vbest,numiter,output]  = SVT(n, trIndex, trData, tsIndex, tsData, tau, delta, maxiter, tol)
%
% Inputs:
%   n               - size of the matrix A assumed n(1) by n(2). 
%   trIndex         - indices set of training entries.
%   trData          - data vector of training set
%   tau             - tau ||A||_* + 0.5 ||A||_F^2
%   delta           - SVT step size
%   maxiter         - maximum number of iterations
%   tol             - stopping criteria (default: 1e-4)
%
% Outputs: completed matrix X stored in SVD format X = U*diag(S)*V' 
%   Ubest           - n1xk left singular vectors 
%   Sbest           - kx1 singular value vector
%   Vbest           - n2xk right singular vectors 
%   numiter         - number of iterations to achieve convergence
%   output          - a structure with data from each iteration.  Includes:
%       output.nuclearNorm      - nuclear norm of current iterate
%       output.rank             - rank of current iterate
%       output.trRes            - error on training set
%       output.tsRes            - error on test set
%
% Algorithm Description: 
% References:
%    Original SVT Algorithm:
%    J. Cai, E. J. Candes, Z. Shen, A singular value thresholding algorithm
%    for matrix completion, SIAM J. Optim., 20(4): 1956-1982, 2010.
%    
%    SVT-R4SVD:
%    Y. Li, W. Yu, A Fast Implementation of Singular Value Thresholding 
%    Algorithm using Recycling Rank Revealing Randomized Singular Value 
%    Decomposition, submitted to Mathematical Program Computation, 2017.

% Written by: Yaohang Li
% Email: yaohang@cs.odu.edu
% Created: Apr. 14, 2017
% 
% The SVT-R4SVD Program is based on the SVT implementation 
% by Emmanuel Candes

%% set parameters
n1 = n(1);                                  % # of rows in the matrix
n2 = n(2);                                  % # of columns in the matrix
m = length(trIndex);                        % # of samples in training set
t = length(tsIndex);                        % # of samples in test set
normTrData = norm(trData);                  % 2-norm of training set
normTsData = norm(tsData);                  % 2-norm of test set
beta = 0.95;                                % simulated annealing cooling factor

%% prepare initial matrix Y(0)
[i, j] = ind2sub([n1,n2], trIndex); 
Y = sparse(i,j,trData,n1,n2,m);             % set up sparse matrix Y
normProjM = normest(Y,1e-2);                % 2-norm of Y
k0 = ceil(tau/(delta*normProjM));           % initialize k0, same as SVT
y = k0*delta*trData;                        % initialize Y(0)
Y = setsparse(Y, i, j, y);                  % sparcify Y(0)         

%% start SVT iterations
r = 0;                                      % initialize rank
U = []; S = []; V = [];                     % initialize U, S, V
Ubest = []; Sbest = []; Vbest = [];         % initialize Ubest, Dbest, Vbest
preU = [];                                  % recycling singular vector (empty)
percent = 0.50;                             % 50% error is allowed at the beginning
mintrRes = 1000000000.0;                    % minimum training set error (set to infinity)
mintsRes = 1000000000.0;                    % minimum test set error (set to infinity)

for k = 1:maxiter
    % estimate the SVT operator
    % perform R4SVD to obtain a low-rank approximation with singular values
    % greater than tau 
    [U,Sigma,V] = r4svd(Y,tau,preU,percent);        % R4SVD
    %preU = U;                                       % recycle SVD
    sigma = diag(Sigma); 
    r = size(Sigma, 1);                             % rank after SVT operator
    Sigma = Sigma(1:r) - tau;                       % subtract tau
    
    % get elements on the sample locations from the completed matrix 
    tr = XonOmega(U*diag(Sigma),V,trIndex);
    ts = XonOmega(U*diag(Sigma),V,tsIndex);
    
    % keep track of norm in 
    trRes = norm(tr - trData)/sqrt(m);
    tsRes = norm(ts - tsData)/sqrt(t);
    
    fprintf('iteration %4d, rank is %2d, training err: %.2e, test err: %.2e\n',k, r, trRes, tsRes);
    
    % if training error is reducing, continue;
    % otherwise trigger simulated annealing cooling scheme
    if trRes < mintrRes
        mintrRes = trRes;
    else
        percent = percent * beta;                   % cooling scheme      
    end
    
    % keep track of the recovery with best test error
    if tsRes < mintsRes
        mintsRes = tsRes;
        Ubest = U; Sbest = Sigma; Vbest = V;
    end
    
    out.trRes(k) = trRes;
    out.tsRes(k) = tsRes;
    out.rank(k) = r;
    out.nuclearNorm(k) = sum(Sigma);

    % check convergence
    if (trRes < tol)
        break
    end
    if (trRes > 1e5)
        disp('Divergence!');
        break
    end
      
    % update matrix Y(i)
    y = y + delta*(trData-tr);
    Y = setsparse(Y, i, j, y);
end

numiter = k;

end