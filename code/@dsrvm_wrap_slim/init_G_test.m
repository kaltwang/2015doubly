function G = init_G_test( obj, G, model_act )

[~, w_idx] = get_wv( model_act, 'w', false);
[~, v_idx] = get_wv( model_act, 'v', false);

if obj.scale_separate
    % scale Gram matrix
    G = bsxfun(@times, G, obj.Scales_w(1,w_idx,1));
    G = bsxfun(@times, G, obj.Scales_v(1,1,v_idx));
else
    % scale Gram matrix
    G = bsxfun(@times, G, obj.Scales_wv(1,w_idx,v_idx));
end

end