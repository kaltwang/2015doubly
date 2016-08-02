function [obj, deltaLogML] = optimize_wv(obj, wv_first, max_count)

logML_before = get_likelihood( obj.model );

if wv_first == 'w'
    wv_second = 'v';
else
    wv_second = 'w';
end

count = 0;
stop_condition = count >= max_count;

logML_all = logML_before;

while ~stop_condition
    count = count+1;
    w_last = obj.model.w;
    v_last = obj.model.v;
    sigma2inv_last = obj.model.sigma2inv;

    logML_0 = get_likelihood( obj.model );
    obj.model = update_wv(obj.model, obj.G, obj.t, wv_first);
    logML_1 = get_likelihood( obj.model );
%    obj = add_history( obj );
    obj.model = update_wv(obj.model, obj.G, obj.t, wv_second);
    logML_2 = get_likelihood( obj.model );
    logML_all(end+1) = logML_2;
%    obj = add_history( obj );

    diff_w = max(abs(w_last - obj.model.w));
    diff_v = max(abs(v_last - obj.model.v));
    diff_s = max(abs(sigma2inv_last - obj.model.sigma2inv));

    % relative difference of the last step in comparison to the overall
    % difference
    diff_logML = logML_2 - logML_0;
    diff_rel_logML = diff_logML / (logML_2 - logML_before);

    stop_diff_wvs = (diff_w < 1e-6) && (diff_v < 1e-6) && (diff_s < 1e-6);
    stop_diff_logML = diff_rel_logML < obj.min_wv_deltaLogML;
    % stop_diff_logML = diff_logML < obj.min_wv_deltaLogML;
    stop_condition = stop_diff_logML || count >= max_count;
end

logML_after = get_likelihood( obj.model );
deltaLogML = logML_after - logML_before;

obj = add_history( obj );

if ~isfinite(deltaLogML)
    deltaLogML = Inf;
end

end