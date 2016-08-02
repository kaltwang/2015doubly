function obj = init_G( obj, G)
% G is NxMxK gram matrix (M==N, if all training samples represent possible basis)

[N, M, K] = size(G); 
obj.G = G;

% ||G_mk||: 1 x M x K
% this would be the norm for each basis and each kernel separately
G_mk_norm = sqrt(sum(obj.G.^2, 1));
% RVM does Scales = G_m_norm which normalizes each of the M basis to
% length 1.
% If we would want to achieve the same, we would need the MxK scales
% defined above, however we only have M+K possibilities, with the scales
% beeing Scale(m,k) = Scales_w * Scales_v'.
% We take the geometric mean (because we want to find acommon scale) along the basis/kernel dimensions and later weight both values by the number of
% dimensions. By doing this, we recover the RVM normalization in case M==1 or K==1.

if obj.scale_separate
    % ignore basis/kernels with norm 0
    G_mk_norm(G_mk_norm == 0) = NaN;

    if M == 1 && K == 1
        % catch division by zero
        weight_w = 0.5;
        weight_v = 0.5;
    else
        weight_w = (M-1)/(M+K-2);
        weight_v = (K-1)/(M+K-2);
    end
    obj.Scales_w = (nangeomean(G_mk_norm,3).^weight_w).^-1;
    obj.Scales_v = (nangeomean(G_mk_norm,2).^weight_v).^-1;

    % avoid NaNs / Infs
    obj.Scales_w(~isfinite(obj.Scales_w)) = 1;
    obj.Scales_v(~isfinite(obj.Scales_v)) = 1;

    % scale Gram matrix
    obj.G = bsxfun(@times, obj.G, obj.Scales_w);
    obj.G = bsxfun(@times, obj.G, obj.Scales_v);
else
    obj.Scales_wv = G_mk_norm.^-1;
    % avoid NaNs / Infs
    obj.Scales_wv(~isfinite(obj.Scales_wv)) = 1;
    % scale Gram matrix
    obj.G = bsxfun(@times, obj.G, obj.Scales_wv);
end

end