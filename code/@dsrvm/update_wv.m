function [obj] = update_wv( obj, G, t, wv)

if wv == 'w'
    PHI = get_Phi(obj, G, false);
    Alpha = obj.alpha;
    comp_S = obj.v_S;
    comp_Mu = obj.v;
    operation_GtG_tprod = [-1 1 2 -2];
    operation_GtG_permute = [1 3 4 2];
    operation_GtG_sum = [1 4];
else
    PHI = get_Psi(obj, G, false);
    Alpha = obj.beta;
    comp_S = obj.w_S;
    comp_Mu = obj.w;
    operation_GtG_tprod = [1 -1 -2 2];
    operation_GtG_permute = [3 1 2 4];
    operation_GtG_sum = [2 3];
end

% (1) w update
% Evtv: KxK (Expected value of v' * v)
Evtv = comp_S + comp_Mu * comp_Mu';
% ESN: EPhitPhi_(m1,m2) = GtG(k1,m1,m2,k2) * Evtv_(k1,k2)
% GtG: KxMxMxK
% EPhitPhi: MxM (Expected value of Phi_v' * Phi_v)

EPhitPhi = tprod(obj.GtG, operation_GtG_tprod, Evtv, [-1 -2]);
% same operation in pure MATLAB, but slower:
% M = size(Alpha,1);
% Evtv = permute(Evtv, operation_GtG_permute);
% EPhitPhi = reshape(sum(sum(bsxfun(@times, obj.GtG, Evtv), operation_GtG_sum(1)), operation_GtG_sum(2)), [M M]);

SIGMAinv = EPhitPhi * obj.sigma2inv + diag(Alpha);

[U, f]		= chol(SIGMAinv);
Ui	= inv(U);
SIGMA	= Ui * Ui';

diagU = diag(U);
if (f ~= 0)|| any(abs(diagU) < eps(max(abs(diagU)))*size(Ui,1))
    error(message('stats:gmdistribution:wdensity:IllCondCov'));
end

% Posterior mean Mu
Mu	= (SIGMA * (PHI'*t)) * obj.sigma2inv;
logdetHOver2 = sum(log(diag(U)));

% (2) sigma^2 update
% we can reuse EPhitPhi for a sigma2inv update:
% Ewtw: MxM (Expected value of w * w')
Ewtw = SIGMA + Mu * Mu';
% ESN: EGtG = EPhitPhi_(m1,m2) * Ewtw_(m1,m2)
% EPhitPhi: MxM (Expected value of Phi_v' * Phi_v)
% EGtG: 1x1 (Expected value of the inner product of G along all dimensions)
EGtG = tprod(EPhitPhi, [-1 -2], Ewtw, [-1 -2]);

% EErr2: 1x1 (Expected squared error value)
% EErr2 = E[ ||t-PHI*Mu||^2 ]
EErr2 = t'*t - 2 * t' * PHI * Mu + EGtG;
% sigma^2 = E[err]^2 / N
N = size(t,1);
obj.sigma2inv = N / EErr2;
obj.N = N;


if wv == 'w'
    obj.w = Mu;
    obj.w_S = SIGMA;
    obj.w_logdetSOver2	= logdetHOver2;
    obj.w_SU = Ui;
else
    obj.v = Mu;
    obj.v_S = SIGMA;
    obj.v_logdetSOver2	= logdetHOver2;
    obj.v_SU = Ui;
end

% update likelihood
[obj, lklhd1] = update_likelihood( obj );
if obj.balance_weights
    % balance weights
    [obj, c] = balance_weights_ab(obj);
    % update likelihood again
    [obj, lklhd2] = update_likelihood( obj );
    % % just to check
    % if abs(lklhd2 - lklhd1) > 1e-3
    %      disp(['c = ' num2str(c) ', diff = ' num2str(lklhd2 - lklhd1)]);
    % end
end

obj.update = wv;
obj.action = NaN;
obj.count_wv = obj.count_wv + 1;
end