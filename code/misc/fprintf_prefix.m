function [ ] = fprintf_prefix( str, formatSpec, varargin )

prefix = ['[' str '] '];
fprintf([prefix formatSpec], varargin{:});

end

