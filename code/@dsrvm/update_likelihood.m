function [obj, logML] = update_likelihood( obj )

% Variational lower bound:
% VarL = E[log( p(t|w,v,sigma^2) )] + E[log( p(w|alpha) )] 
%     + E[log( p(v|beta))] - E[log( q(w) )] - E[log( q(v) )]

EData = (obj.N*log(obj.sigma2inv))/2; % E[log( p(t|w,v,sigma^2) )]
obj.EpAlpha = sum(log(obj.alpha))/2 - (obj.w.^2 + diag(obj.w_S))' * obj.alpha/2; % E[log( p(w|alpha) )]
obj.EpBeta = sum(log(obj.beta))/2 - (obj.v.^2 + diag(obj.v_S))' * obj.beta/2; % E[log( p(v|beta))]
VarL = EData + obj.EpAlpha + obj.EpBeta ... 
    - obj.w_logdetSOver2 ... % - E[log( q(w) )]
    - obj.v_logdetSOver2; % - E[log( q(v) )]

logML = VarL;

if isempty(logML)
    logML = -Inf;
end

obj.logML = logML;

end