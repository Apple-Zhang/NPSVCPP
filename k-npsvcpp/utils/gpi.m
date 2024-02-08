function [W, alpha] = gpi(A, B, W0)
%GPI Generalized Power Iteration
%   W = gpi(A, B, W0)
%   
%   See the paper by Nie et. al.: https://arxiv.org/pdf/1701.00381.pdf
%   Written by Apple Zhang, 2024.
%
MAX_ITER = 25;

W = W0;
m = size(A, 1);
alpha = eigs(A, 1) + .1;
A = alpha*eye(m) - A;

for kk = 1:MAX_ITER
    M = A*W + B;
    [U, ~, V] = svd(M, 'econ');
    W = U*V';
end
end