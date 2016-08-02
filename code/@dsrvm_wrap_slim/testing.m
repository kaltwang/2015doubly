function t = testing( obj, G)

model_act = obj.model_best;

G = init_G_test( obj, G, model_act );
t = predict(model_act, G, false);

% re-scale targets to original space
t = normalize_mean_std_inv( t, obj.t_mean, obj.t_std );

end