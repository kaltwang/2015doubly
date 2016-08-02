%% Create data
n_samples = 1000; a = 2; b = 1; c = 0.1; noise = 0.3;
seed = 0;
[ X_train, t_train ] = make_sonnenburg_data( n_samples, a, b, c, noise, seed);
seed = 1234;
[ X_test, t_test ] = make_sonnenburg_data( n_samples, a, b, c, noise, seed);


%% Create the dsrvm algorithm and set parameters
% (See dsrvm_wrap_slim.m for all parameters.)
algo = dsrvm_wrap_slim(...
    'plot',1,...
    'max_wv_update', 20,...
    'max_iterations', 500);


%% Training
% Get the NxMxK kernel gram matrix for the training data
% (all training samples are used as possible basis functions and thus
% N==M).
% In this example, we use 10 gaussian kernels, each one with a different 
% basis width
basisWidth = [0.001 0.005 0.01 0.05 0.1 1 10 50 100 1000];
G_train = kernel_gauss(X_train, X_train, basisWidth);

% perform training
algo_t = training(algo,G_train,t_train);


%% Testing
% get the indices of relevant vectors and relevant kernels
[w_idx, v_idx] = get_wv_idx(algo_t);
RV = X_train(w_idx,:);
RK = basisWidth(v_idx);
% the test Gram matrix uses only relevance vectors and relevant kernels
G_test = kernel_gauss(X_test, RV, RK);
% perform testing
pred_test = testing(algo_t,G_test);

%% Get evaluation measures
MSE = mean((pred_test-t_test).^2);
num_RV = get_num_RV(algo_t);
num_RK = get_num_RK(algo_t);
num_iter = algo_t.model_best.iter;
lklhd = algo_t.model_best.logML;
error_stdev = sqrt(algo_t.model_best.sigma2inv^-1);
fprintf_prefix(mfilename, 'MSE=%.3f, #RV=%d, #RK=%d, #iter=%d, lklhd=%.3f, error_stdev=%.3f \n', MSE, num_RV, num_RK, num_iter, lklhd, error_stdev);


%% Visualize Results
% plot the fitted curve
figure;
plot(X_train, t_train);
hold on;
plot(X_test, pred_test, 'r', 'LineWidth', 2);

% plot the RV's
scatter(RV, t_train(w_idx,:), 'LineWidth', 2, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'black');
xlabel('Features (X)');
ylabel('Targets (t)');
legend('Training input', 'Fittet function t = y(X)','Relevance vectors (RV)', 'Location','northwest');