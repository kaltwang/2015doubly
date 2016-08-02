function [val, idx] = get_wv( obj, wv, full)

if ~exist('full','var') || isempty(full)
    full = false;
end

num_obj = numel(obj);
idx_all = cell2mat(arrayfun(@(x) x.([wv '_idx']), obj, 'UniformOutput', false));
assert(islogical(idx_all));
if full
    idx = true(1,size(idx_all,2));
else
    idx = any(idx_all,1);
end
num_idx = sum(idx);
val = zeros(num_obj, num_idx);

for i = 1:num_obj
    obj_act = obj(i);
    idx_all_act = idx_all(i, idx);
    val(i, idx_all_act) = obj_act.(wv);
end

end