function [ X, x_mean, x_std] = normalize_mean_std( X, x_mean, x_std )
% Normalize each column of X

if iscell(X)
    if ~exist('x_mean','var') || isempty(x_mean);
        x_mean = repmat({[]}, size(X));
    end
    if ~exist('x_std','var') || isempty(x_std);
        x_std = repmat({[]}, size(X));
    end
    [ X, x_mean, x_std] = cellfun( @normalize_mean_std, X, x_mean, x_std, 'Uni', false );
else

    if ~exist('x_mean','var') || isempty(x_mean);
        x_mean = mean(X,1);
    end

    if ~exist('x_std','var') || isempty(x_std);
        x_std = std(X,[],1);
    end

    x_std_inv = x_std.^-1;
    x_std_inv(~isfinite(x_std_inv)) = 1;

    X = bsxfun(@minus, X, x_mean);
    X = bsxfun(@times, X, x_std_inv);

end

end

