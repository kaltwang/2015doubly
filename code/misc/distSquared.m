function D = distSquared(X,Y)
% X: N x F data matrix
% Y: M x F data matrix
% returns:
% D: N x M pairwise euclidean distances
% i.e., D(n,m) = (X(n,:) - Y(m,:))^2 for all n,m

 D = -2*X*Y';
 D = bsxfun(@plus, D, sum((X.^2),2));
 D = bsxfun(@plus, D, sum((Y.^2),2)');

end