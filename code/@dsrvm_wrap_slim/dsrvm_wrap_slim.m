
classdef dsrvm_wrap_slim

properties
    % targets: N x 1
    t = [];
    % Kernel gram matrix: N x M x K
    G = [];
    
    % parameters
    plot = false; % visualization
    scale_separate = false; % scale w/v separately or together
    
    % convergence parameters for inner loop (w/v updates)
    max_wv_update = 10;
    min_wv_deltaLogML = 1e-5;
    
    % convergence parameters for outer loop  (a, w/v, b, w/v updates)
    max_iterations = 1000;
    max_fail = 5;
    min_deltaLogML = 1e-6;

    % Normalization values
    Scales_w = [];
    Scales_v = [];
    Scales_wv = [];   
    t_mean = [];
    t_std = [];  
    
    % current model
    model = dsrvm();
    % best model
    model_best = dsrvm.empty();
    % history of models
    hist_model = dsrvm.empty();
end

methods
    % constructor
    function obj = dsrvm_wrap_slim(varargin)               
        obj = set(obj,varargin{:});
    end
    
    obj = set( obj, varargin );
    
    % initializes X, t and G
    obj = init_G( obj, G);
    obj = optimize_model( obj );
    [obj, deltaLogML] = optimize_wv(obj, wv_first, max_count);
    G = init_G_test( obj, G, model_act );
    
    d_test = testing( obj, d_test, d_train);
    
    function [obj] = training( obj, G, t)
        % scale targets
        [ obj.t, obj.t_mean, obj.t_std] = normalize_mean_std( t );
        % scale kernel gram matrix
        obj = init_G( obj, G);
        % initialize model parameters
        obj.model = init( obj.model, obj.G, obj.t);
        % optimize the model
        obj = optimize_model( obj );
        % clear data
        obj.G = [];
    end

    function k = get_kernel( obj )
        k = obj.kernel;
    end

    function obj = set_kernel( obj, k)
        obj.kernel = k;
    end
    
    function obj = add_history( obj )
        m = clear(obj.model);
        obj.hist_model(end+1,1) = update_error(m, obj.G, obj.t);
        % reset count_wv
        obj.model.count_wv = 0;
    end
    
    function M = get_num_RV( obj )
        [M, ~] = get_MK(obj.model_best);
    end
    function K = get_num_RK( obj )
        [~, K] = get_MK(obj.model_best);
    end
    function kw = get_kernel_weights( obj )
        kw = shiftdim(get_wv(obj.model_best, 'v', true));
    end
    
    function [w_idx, v_idx] = get_wv_idx(obj)
        [~, w_idx] = get_wv( obj.model_best, 'w', false);
        [~, v_idx] = get_wv( obj.model_best, 'v', false);
    end
end % methods
end % classdef
