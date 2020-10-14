%% main program for using SVT-R4SVD for image recovery
% 
% Written by: Yaohang Li
% Email: yaohang@cs.odu.edu
% Created: Apr. 14, 2017

clear all
close all
clc

addpath('setsparse');
rng('default')
format long

%% Load the original matrix from the image
% M3 = imread('..\images\juliabw.jpg');
% M3 = imread('..\images\lena_1024.jpg');
M3 = imread('..\images\baboon.gif');
M = double(M3(:,:,1));

% Size of the matrix
[n1,n2] = size(M);

%% Set parameters
maxiter = 200;             % maximun number of iterations in SVT
tol = 1.0e-00;              % convergence threshold of SVT
trpercent = 0.20;           % specifying percentage of training samples
tspercent = 0.05;           % specifying percentage of test samples

%% generate random samples from the loaded image
m = floor((n1*n2)*trpercent);           % number of training samples
t = floor((n1*n2)*tspercent);           % number of test samples
reorder = randsample(n1*n2,n1*n2);      % reorder the samples 
trIndex = reorder(1:m);                 % an array of training sample indices
tsIndex = reorder(m+1:m+t);             % an array of test sample indices
trData = M(trIndex);                    % an array of training samples
tsData = M(tsIndex);                    % an array of test samples

%% SVT parameters
tau = norm(trData, 'fro')*sqrt(n1*n2/m);
delta = sqrt(n1*n2/m);

disp(['number of rows in image (n1): ',num2str(n1)])
disp(['number of columns in image (n2): ',num2str(n2)])
disp(['regularization parameter (tau): ',num2str(tau)])
disp(['percentage of training samples (percent): ',num2str(trpercent)])
disp(['step size (delta): ',num2str(delta)])
disp(['convergence threshold of SVT (tol): ',num2str(tol)])
disp(['maximun number of iterations in SVT (maxiter): ',num2str(maxiter)])

%% The modified SVT algorithm based on R3SVD
fprintf('\nSolving by SVT using R4SVD...\n');
% create the sample matrix for visualization
[x, y] = ind2sub([n1,n2], trIndex);
T = ones(n1,n2)*200;
for i = 1:m
    T(x(i),y(i)) = M(x(i),y(i));
end
figure(1)
imshow(uint8(T)); title([num2str(trpercent*100),'% training Samples'])
[U2,S2,V2,numiter2,out] = SVT_r4svd([n1 n2],trIndex,trData,tsIndex,tsData,tau,delta,maxiter,tol);

% construct the completed matrix
X2 = U2*diag(S2)*V2';
% Show results
fprintf('RSVD: The recovered rank is %d\n',rank(X2) );
fprintf('RSVD: The relative recovery error is: %d\n', norm(M-X2,'fro')^2/norm(M,'fro')^2)

% Results display
figure(2);
subplot(1,3,1)
imshow(uint8(M)); title('Original')
subplot(1,3,2)
imshow(uint8(T)); title([num2str(trpercent*100),'% training Samples'])
subplot(1,3,3)
imshow(uint8(X2)); title('SVT using R3SVD')

figure(3);
semilogy(out.rank, out.trRes, out.rank, out.tsRes);
xlabel('Rank'); ylabel('Residual Error'); legend('Training Set','Test Set');


