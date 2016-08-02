function [Psi] = get_Psi( obj, G, full )

if ~exist('full','var') || isempty(full)
    full = true;
end

[N, ~, K] = size(G);
M = size(obj.w,1);

% G: N x M x K
% v: K x 1
% Phi: N x M
if full
    Psi = reshape(reshape(permute(G(:,obj.w_idx,:), [1 3 2]), N*K, M) * obj.w, N, K);
else
    K = size(obj.v,1);
    Psi = reshape(reshape(permute(G(:,obj.w_idx,obj.v_idx), [1 3 2]), N*K, M) * obj.w, N, K);
end

end

