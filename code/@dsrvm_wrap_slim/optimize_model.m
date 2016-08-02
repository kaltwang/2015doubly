function obj = optimize_model( obj )

ACTION_TERMINATE = 10;
% add initial model to the history
obj = add_history( obj );

[M, K] = get_MK( obj.model );
if M >= K
    wv_first = 'w';
    ab_first = 'a';
    wv_second = 'v';
    ab_second = 'b';
else
    wv_first = 'v';
    ab_first = 'b';
    wv_second = 'w';
    ab_second = 'a';
end
[obj, deltaLogML] = optimize_wv(obj, wv_first, 10);
%[obj.model, deltaLogML_w] = update_wv(obj.model, obj.G, obj.t, wv_first);
%[obj.model, deltaLogML_v] = update_wv(obj.model, obj.G, obj.t, wv_second);
logML_best = get_likelihood( obj.model );
obj.model_best = clear(obj.model);

stop_condition = false;
obj.model.iter = 0;
fail = 0;
print_next = true;
while ~stop_condition
    obj.model.iter = obj.model.iter + 1;
    [obj.model, selectedAction_a, deltaLogML_a] = update_ab(obj.model, obj.G, obj.t, ab_first);
    obj = add_history( obj );
    [obj, deltaLogML_w] = optimize_wv(obj, wv_second, obj.max_wv_update);
    [obj.model, selectedAction_b, deltaLogML_b] = update_ab(obj.model, obj.G, obj.t, ab_second);
    obj = add_history( obj );
    [obj, deltaLogML_v] = optimize_wv(obj, wv_first, obj.max_wv_update);

    logML_act = get_likelihood( obj.model );
    deltaLogML = logML_act - logML_best;
    if deltaLogML > 0
        logML_best = logML_act;
        obj.model_best = clear(obj.model);
    end

    iter = obj.model.iter;
    stop_LogML = deltaLogML < obj.min_deltaLogML;
    if stop_LogML
        fail = fail + 1;
        fprintf_prefix(mfilename, ...
            'iter=%d, deltaLogML=%.3f < min_deltaLogML=%.3f\n',...
            iter, deltaLogML, obj.min_deltaLogML)
        print_next = true;
    else
        fail = 0;
    end
    stop_fail = fail >= obj.max_fail;
    if stop_fail
        fprintf_prefix(mfilename, ...
            'iter=%d, stopping because fail >= max_fail == %d\n',...
            iter, obj.max_fail)
        print_next = true;
    end
    
    stop_Iteration = obj.model.iter >= obj.max_iterations;
    if stop_Iteration
        fprintf_prefix(mfilename, ...
            'iter=%d, stopping because iter >= max_iterations == %d\n',...
            iter, obj.max_iterations)
        print_next = true;
    end
    
    stop_Action = selectedAction_a == ACTION_TERMINATE && selectedAction_b == ACTION_TERMINATE;
    if stop_Action
        fprintf_prefix(mfilename, ...
            'iter=%d, (could be stopping) selectedAction_a and selectedAction_b are both ACTION_TERMINATE\n', ...
            iter)
        print_next = true;
    end
    
    stop_condition = stop_fail || stop_Iteration;

    if obj.plot
        plot(obj.hist_model);
    end
    
    if print_next
        fprintf_prefix(mfilename, ...
            'iter=%d, logML=%.3f, d_logML=%.3f (d_logML_a=%.3f, d_logML_w=%.3f, d_logML_b=%.3f, d_logML_v=%.3f)\n', ...
            iter, logML_act, deltaLogML, deltaLogML_a, deltaLogML_w, deltaLogML_b, deltaLogML_v)
        print_next = false;
    end
end

end