function G = init_data_test( obj, X_test, model_act )

[~, w_idx] = get_wv( model_act, 'w', false);
[~, v_idx] = get_wv( model_act, 'v', false);

if isa(obj.kernel, 'kernel_cache')
    obj.kernel = init_cache(obj.kernel);
end

% G is NxMxK gram matrix (it should be M==N)
% [ obj.G, obj.kernel ] = get_F( obj.kernel, X, 1:M, 1:K);
feature_rel = find(v_idx);
RV = obj.X(w_idx,:);
[ G, obj.kernel ] = get_dist_tensor( obj.kernel, X_test, feature_rel, RV );

% clear kernel cache
if isa(obj.kernel, 'kernel_cache')
    obj.kernel = clear_cache(obj.kernel);
end

G = init_G_test( obj, G, model_act );

end