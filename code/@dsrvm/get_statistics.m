function s = get_statistics( obj, sname)

% EData = (obj.N*log(obj.sigma2inv))/2; % E[log( p(t|w,v,sigma^2) )]
% obj.EpAlpha = sum(log(obj.alpha))/2 - (obj.w.^2 + diag(obj.w_S))' * obj.alpha/2; % E[log( p(w|alpha) )]
% obj.EpBeta = sum(log(obj.beta))/2 - (obj.v.^2 + diag(obj.v_S))' * obj.beta/2; % E[log( p(v|beta))]
% VarL = EData + obj.EpAlpha + obj.EpBeta ... 
%     - obj.w_logdetSOver2 ... % - E[log( q(w) )]
%     - obj.v_logdetSOver2; % - E[log( q(v) )]

% needed for VarL:
% 1           2      3        4       5               6               7      8      9       10 11 12   13    14
% diff logML, logML, EpAlpha, EpBeta, w_logdetSOver2, v_logdetSOver2, EData, sigma, action, M, K, MSE, CORR, count_wv

if ~exist('sname','var') || isempty(sname)
    sname = 'likelihood';
end

num_obj = length(obj);

switch sname
    case 'likelihood'
        fh = @(x) [...      % 1 (diff logML), 2 (logML)
            x.EpAlpha ...   % 3
            x.EpBeta ...    % 4
            x.w_logdetSOver2 ... % 5
            x.v_logdetSOver2 ... % 6
            (x.N*log(x.sigma2inv))/2 ... % 7 (EData)
            sqrt(x.sigma2inv^-1) ... % 8 (sigma)
            x.action ...    % 9
            length(x.w) ... % 10 (M)
            length(x.v) ... % 11 (K)
            x.MSE ...       % 12
            x.CORR ...      % 13
            x.count_wv ...  % 14
            x.iter ...      % 15
            ];
        s = zeros(num_obj,15);
        s(:,3:end) = cell2mat(arrayfun(fh, obj, 'UniformOutput', false));
        s(:,2) = s(:,7) + s(:,3) + s(:,4) - s(:,5) - s(:,6); % 2
        s(2:end,1) = diff(s(:,2)); % 1

    case {'w','v'}
        s = zeros(num_obj, length(obj(1).([sname '_idx'])));
        for i = 1:num_obj
            obj_act = obj(i);
            s(i, obj_act.([sname '_idx'])) = obj_act.(sname);
        end
end

end