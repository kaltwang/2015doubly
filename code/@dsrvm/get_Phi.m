function [Phi] = get_Phi( obj, G, full )

if ~exist('full','var') || isempty(full)
    full = true;
end

[N, M, ~] = size(G);
K = size(obj.v,1);

% ESN: Phi_(n,m) = G_(n,m,k) * v_(k)
% G: N x M x K
% v: K x 1
% Phi: N x M
if full
    if N <= 2000
        Phi = reshape(reshape(G(:,:,obj.v_idx), N*M, K) * obj.v, N, M);
    else
        % K relatively low, therefore the loop saves a lot of memory:
        Phi = zeros(N,M);
        v_idx_lin = find(obj.v_idx);
        for k = 1:K
            Phi = Phi + G(:,:,v_idx_lin(k)) * obj.v(k);
        end
    end
else
    M = size(obj.w,1);
    Phi = reshape(reshape(G(:,obj.w_idx,obj.v_idx), N*M, K) * obj.v, N, M);
end

end

