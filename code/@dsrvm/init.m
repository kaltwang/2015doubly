function obj = init( obj, G, t)

[N, M, K] = size(G);

% init sigma^2
% initialize sigma^2 with the variance of the targets
stdt = max([1e-6 std(t, [], 1)]);
obj.sigma2inv = (1/stdt)^2;
obj.N = N;

% init w_idx and v_idx
% Set initial basis m and kernel k to have the largest projection on the
% targets normalized by ||G_mk||, i.e.
% max_mk (|G_mk * t| / ||G_mk||)
% G: N x M x K; t: N x 1
G_n_mk = reshape(G, N, M*K);

% G_mk_t: M x K
G_mk_t = reshape(t' * G_n_mk, M, K);
% ||G_mk||: M x K
G_mk_norm = reshape(sqrt(sum(G.^2,1)), M, K);

G_mk_t_norm = G_mk_t ./ G_mk_norm;

if obj.init_w_single && obj.init_v_single
    % select the max of all basis/kernels
    [max_ms, ms] = max(abs(G_mk_t_norm),[],1);
    [~, k] = max(max_ms,[],2);
    m = ms(k);
end
if obj.init_w_single && ~obj.init_v_single
    % select the max basis of the mean of the kernels and all kernels
    k = 1:K;
    [~, m] = max(mean(abs(G_mk_t_norm),2),[],1);
end
if ~obj.init_w_single && obj.init_v_single
    % select the max kernel of the mean of the basis and all basis
    m = 1:M;
    [~, k] = max(mean(abs(G_mk_t_norm),1),[],2);
end
if ~obj.init_w_single && ~obj.init_v_single
    % select all basis and all kernels
    m = 1:M;
    k = 1:K;
end

if obj.init_rand
    % use only the upper 50% of aligned basis
    num_select = ceil(M*K/2);
    [~, idx] = sort(abs(G_mk_t_norm(:)), 'descend');
    idx = idx(1:num_select);
    tmp = abs(G_mk_t_norm(idx));
    
    % exclude values below 1.01
    idx2 = tmp > 1.01;
    idx = idx(idx2);
    
    seed_old = rng(obj.init_seed);
%     m = randperm(M,1);
%     k = randperm(K,1);
    sel_idx = randperm(length(idx),1);
    rng(seed_old);
    sel = idx(sel_idx);
    [m, k] = ind2sub([M K], sel);
end

if obj.init_rvm
    likelihood_ = 'Gaussian';
    % basis is the convex combination of all kernels with equal weights
    BASIS = mean(G,3);
    Targets = t;
    OPTIONS	= SB2_UserOptions('iterations',1000,'diagnosticLevel', 2, 'monitor', 50);
    % Note: the SparseBayes toolbox must be available
    %(download from http://www.miketipping.com/)
    [PARAMETER, HYPERPARAMETER, DIAGNOSTIC, BASIS] = SparseBayes(likelihood_, BASIS, Targets, OPTIONS);
    m = PARAMETER.Relevant';
    k = 1:K;
    w = PARAMETER.Value;
    w_S = PARAMETER.Sigma;
end

obj.w_idx = false(1,M);
obj.v_idx = false(1,K);
obj.w_idx(m) = true;
obj.v_idx(k) = true;
% init w/v with normalized unit vector (only effectively used when init full w/v)
obj.w = ones(length(m),1) / length(m);
obj.v = ones(length(k),1) / length(k);
% initialize covariances with ones (will be updated later)
if obj.init_rvm
    obj.v_S = eye(length(k));
    obj.w_S = eye(length(m));
else
    obj.w_S = eye(length(m));
    obj.v_S = eye(length(k));
end

obj.w_logdetSOver2 = length(m);
obj.v_logdetSOver2 = length(k);
    

% exact alpha/beta in init_w_single && obj.init_v_single case,
% otherwise
% heuristic alpha/beta

% "Reasonable" initial alpha bounds
INIT_ALPHA_MAX	= 1e3;

% PHI: N x M x 1
PHI = get_Phi(obj,G);
PHI = PHI(:,obj.w_idx);
p		= diag(PHI' * PHI) * obj.sigma2inv;
q		= (PHI' * t) * obj.sigma2inv;
alpha	= p.^2./(q.^2-p);
% The main algorithm will handle these automatically shortly
% (i.e. prune them)
alpha(alpha<0) = INIT_ALPHA_MAX; 
obj.alpha = alpha;

% PSI: N x K x 1
PSI = get_Psi(obj,G);
PSI = PSI(:,obj.v_idx);
p		= diag(PSI' * PSI) * obj.sigma2inv;
q		= (PSI' * t) * obj.sigma2inv;
beta	= p.^2./(q.^2-p);
% The main algorithm will handle these automatically shortly
% (i.e. prune them)
beta(beta<0) = INIT_ALPHA_MAX;
obj.beta = beta;

% update the GtG cache
obj = update_GtG( obj, G);

% statistics
obj = update_likelihood( obj );
obj.action = NaN;
obj.update = 'i';
end