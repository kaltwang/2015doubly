
classdef dsrvm

properties
    % parameters
    init_w_single = true;
    init_v_single = true;
    % if both are set, then deletion takes priority over addition
    PriorityAddition = false;
    PriorityDeletion = true;
    % w/v weight balancing
    balance_weights = true;

    % model
    alpha = [];
    w = [];
    w_S = [];
    w_idx = [];
    
    beta = [];
    v = [];
    v_S = [];
    v_idx = [];
    
    % the same as beta in RVM
    sigma2inv = [];
    N = [];
    
    % cached values
    % inner product of G along dim N: KxMxMxK (only active
    % dimensions)
    GtG = [];
    w_logdetSOver2 = []; % 0.5 * log(det(w_S^-1));
    v_logdetSOver2 = []; % 0.5 * log(det(v_S^-1));
    w_SU = []; % w_S = w_SU * w_SU'
    v_SU = []; % v_S = v_SU * v_SU'

    % statistics
    logML = [];
    EpAlpha = [];
    EpBeta = [];
    action = [];
    update = [];
    iter = 0;
    count_wv = 0;
    
    MSE = [];
    CORR = [];
    
    % random initialization
    init_rand = true;
    init_seed = 1234;
    
    % rvm initialization
    init_rvm = false;
end

methods
    % constructor
    function obj = dsrvm(varargin)               
        obj = set(obj,varargin{:});
    end
    
    obj = set( obj, varargin );
    
    % init sigma2, alpha and beta
    obj = init( obj, G, t);
    % update functions
    [obj] = update_wv( obj, G, t, wv);
    [obj, selectedAction, deltaLogML] = update_ab( obj, G, t, ab);
    [obj, logML] = update_likelihood( obj );
    
    % G: N x M x K
    Phi = get_Phi( obj, G, full );
    Psi = get_Psi( obj, G, full );

    % get weights
    [val, idx] = get_wv( obj, wv, full);
    % statistics
    s = get_statistics( obj, sname);
    [] = plot(obj);
    % prediction
    t = predict( obj, G, full );
    
    % w/v balancing
    [obj, c] = balance_weights_ab( obj );
    
    % needs to be called if w_idx/v_idx changed
    % better implementation: incremental updates
    function obj = update_GtG( obj, G)
        % ESN: GtG_(k1,m1,m2,k2) = G_(n,m1,k1) * G_(n,m2,k2)
        %obj.GtG = tprod(G(:,obj.w_idx,obj.v_idx), [-1 2 1], G(:,obj.w_idx,obj.v_idx), [-1 3 4]);
        N = size(G,1);
        M = sum(obj.w_idx);
        K = sum(obj.v_idx);
        G_n_mk = reshape(G(:,obj.w_idx,obj.v_idx), N, M*K);
        obj.GtG = permute(reshape(G_n_mk' * G_n_mk, [M K M K]), [2 1 3 4]);
    end

    function [M, K] = get_MK( obj )
        M = size(obj.w,1);
        K = size(obj.v,1);
    end

    function obj = clear( obj )
        % clear large variables not needed for inference
        obj.w_S = [];
        obj.v_S = [];
        obj.GtG = [];
        obj.w_SU = [];
        obj.v_SU = [];
    end

    function logML = get_likelihood( obj )
        logML = obj.logML;
    end
    
    function obj = update_error( obj, G, t )
        t_pred = predict(obj, G, true);
        obj.CORR = corr(t, t_pred);
        obj.MSE = mean((t - t_pred).^2);
    end  
end % methods
end % classdef
