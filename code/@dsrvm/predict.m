function t = predict( obj, G, full )
num_obj = numel(obj);
N = size(G,1);

if num_obj == 1
    [M, K] = get_MK( obj );

    if full
        G = G(:,obj.w_idx,obj.v_idx);
    else
        assert(size(G,2) == M);
        assert(size(G,3) == K);
    end
    Phi = reshape(reshape(G, N*M, K) * obj.v, N, M);
    t = Phi * obj.w;
else
    % vectorized prediction
    [w, w_idx] = get_wv(obj, 'w', 0);
    [v, v_idx] = get_wv(obj, 'v', 0);
    M = size(w,2);
    K = size(v,2);
    
    if full
        G = G(:,w_idx,v_idx);
    else
        assert(size(G,2) == M);
        assert(size(G,3) == K);
    end
    t = zeros(N, num_obj);
    G = reshape(G, N*M, K);
    for i = 1:num_obj
        Phi = reshape(G * v(i,:)', N, M);
        t(:,i) = Phi * w(i,:)';
    end
end

end
