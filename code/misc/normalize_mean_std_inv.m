function [ X ] = normalize_mean_std_inv( X, x_mean, x_std )
% Inverse of normalize_mean_std

if iscell(X)
    if ~exist('x_mean','var') || isempty(x_mean);
        x_mean = repmat({[]}, size(X));
    end
    if ~exist('x_std','var') || isempty(x_std);
        x_std = repmat({[]}, size(X));
    end
    X = cellfun( @normalize_mean_std_inv, X, x_mean, x_std, 'Uni', false );
else

    if ~exist('x_mean','var') || isempty(x_mean);
        x_mean = zeros(1,size(X,2));
    end

    if ~exist('x_std','var') || isempty(x_std);
        x_std = ones(1,size(X,2));
    end

    x_std_inv = x_std.^-1;
    x_std_inv(~isfinite(x_std_inv)) = 1;

    X = bsxfun(@rdivide, X, x_std_inv);
    X = bsxfun(@plus, X, x_mean);
end

end

