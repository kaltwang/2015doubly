function [ x, y ] = make_sonnenburg_data( n_samples, a, b, c, noise, seed)
% Create artificial data to test regression models

% from
% Sonnenburg, S., Rätsch, G., Schäfer, C., & Schölkopf, B. (2006).
% Large Scale Multiple Kernel Learning.
% J. Mach. Learn. Res., 7, 1531–1565.

rng('default');
rng('shuffle');

if nargin > 5
    rng(seed);
end

x = rand(n_samples,1)*20;
x = sort(x,1,'ascend'); % sort for nicer plotting
y = sin(a * x) + sin(b * x) + c * x + (noise*randn(n_samples,1));

end

