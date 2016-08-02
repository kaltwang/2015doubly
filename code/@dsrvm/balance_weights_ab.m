function [obj, c] = balance_weights_ab(obj)

[M, K] = get_MK( obj );

c = sqrt((norm(obj.w) / norm(obj.v)));

if isnan(c) || isinf(c) % || (p_w == 0) || (p_v == 0)
    c = 1;
end

if c ~= 1
    obj.alpha = obj.alpha * c^2;
    obj.w = obj.w * c^(-1);
    obj.w_S = obj.w_S * c^(-2);
    obj.w_logdetSOver2 = obj.w_logdetSOver2 + M * log(c);
    obj.w_SU = obj.w_SU * c^(-1);
    
    obj.beta = obj.beta * c^(-2);
    obj.v = obj.v * c;
    obj.v_S = obj.v_S * c^2;
    obj.v_logdetSOver2 = obj.v_logdetSOver2 - K * log(c);
    obj.w_SU = obj.w_SU * c;
end

end