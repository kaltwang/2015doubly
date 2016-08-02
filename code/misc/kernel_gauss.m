function G = kernel_gauss(X, Y, basis_width)
% X: N x F data matrix
% Y: M x F data matrix
% basis_width: kernel parameter. If numel(basis_width)>1, then a
% multikernel is konstructed with K=numel(basis_width)
% returns:
% G: N x M x K kernel gram matrix

% first, flatten the basis_width vector and move it to the 3rd dimension
basis_width = shiftdim(basis_width(:),-2);
% get the pairwise euclidean distances
D = distSquared(X, Y);
% scale by the basis width(s)
D_scaled = bsxfun(@rdivide, D, basis_width);
% apply the negative exponential
G = exp(-D_scaled);

end