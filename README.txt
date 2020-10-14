README: SVT-R4SVD algorithm
This package contains the example of using SVT-R4SVD to recover an image with missing pixels. The programs are written in Matlab R2016a.

example_imagerecovery_SVTr4svd.m:       main  program
SVT_r4svd.m:                            SVT   program
r4svd.m:                                r4svd program
baboon.gif:                             example image

Package used:
setsparse package: Retrieve and assign values of sparse matrix in one shot. (by Bruno Luong)
https://www.mathworks.com/matlabcentral/fileexchange/23488-sparse-sub-access?focused=5170479&tab=function						

References:
    Original SVT Algorithm:
    J. Cai, E. J. Candes, Z. Shen, A singular value thresholding algorithm
    for matrix completion, SIAM J. Optim., 20(4): 1956-1982, 2010.
    
    SVT-R4SVD:
    Y. Li, W. Yu, A Fast Implementation of Singular Value Thresholding 
    Algorithm using Recycling Rank Revealing Randomized Singular Value 
    Decomposition, arXiv:1704.05528, 2017.

Written by: Yaohang Li
Email: yaohang@cs.odu.edu
Created: Apr. 14, 2017
