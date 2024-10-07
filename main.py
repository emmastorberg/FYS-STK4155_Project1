import numpy as np

from LinearRegression import BaseModel, OrdinaryLeastSquares, RidgeRegression, LassoRegression
import results as res
import plotroutines as plot

""" Code for all figures in report, and some more."""

n_F = 15 # 30
n_T = 16
seed = 29 # 55555
maxdegree = 8 # 10
multidim = True
scale = True
with_std = False
test_size = 0.4

terrain_file = "datasets/SRTM_data_Norway_1.tif"
terrain_start = 0
terrain_step = 4 #10

x_Franke, y_Franke, = res.generate_data_Franke(n_F, seed, multidim=multidim)
x_terrain, y_terrain = res.generate_data_terrain(n_T, terrain_start, terrain_step, terrain_file)

# ----------------------------------------------- Figure 1, 2 ------------------------------------------------

plot.plot_Franke_function(n_F, noise=False, filename="Figure1a.pdf")
plot.plot_Franke_function(n_F, noise=True, filename="Figure1b.pdf")
plot.plot_terrain(n_T, terrain_start, terrain_step, terrain_file=terrain_file, filename="Figure2.pdf")

# ------------------------------------------- Figure 3a, 3b, 4a, 4b ------------------------------------------

ols_F = res.Results(
    OrdinaryLeastSquares, 
    x_Franke, 
    y_Franke, 
    maxdegree, 
    multidim=multidim, 
    scale=scale, 
    with_std=with_std, 
    test_size=test_size
    )
ols_F.train_and_predict_all_models()
ols_F.calculate_MSE_across_degrees()
ols_F.calculate_R2_across_degrees()

ols_T = res.Results(
    OrdinaryLeastSquares, 
    x_terrain, 
    y_terrain, 
    maxdegree, 
    multidim=multidim, 
    scale=scale, 
    with_std=with_std, 
    test_size=test_size
    )
ols_T.train_and_predict_all_models()
ols_T.calculate_MSE_across_degrees()
ols_T.calculate_R2_across_degrees()

high_n_F = int(n_F * 4)
high_n_T = int(n_T * 4)
low_terrain_step = int(terrain_step / 4)
x_F_high_n, y_F_high_n, = res.generate_data_Franke(high_n_F, seed, multidim=multidim)
x_T_high_n, y_T_high_n = res.generate_data_terrain(high_n_T, terrain_start, low_terrain_step, terrain_file)

ols_F_good_pred = res.Results(
    OrdinaryLeastSquares, 
    x_F_high_n, 
    y_F_high_n, 
    5, 
    multidim=multidim, 
    scale=scale, 
    with_std=with_std, 
    test_size=test_size
    )
ols_F_good_pred.train_and_predict_all_models()
ols_F_good_pred.calculate_MSE_across_degrees()
ols_F_good_pred.calculate_R2_across_degrees()

ols_T_good_pred = res.Results(
    OrdinaryLeastSquares, 
    x_T_high_n, 
    y_T_high_n, 
    5, 
    multidim=multidim, 
    scale=scale, 
    with_std=with_std, 
    test_size=test_size
    )
ols_T_good_pred.train_and_predict_all_models()
ols_T_good_pred.calculate_MSE_across_degrees()
ols_T_good_pred.calculate_R2_across_degrees()

plot.plot_mse_per_polydegree_OLS(ols_F_good_pred, ols_T_good_pred, filename="Figure3a.pdf")
plot.plot_r2_score_per_polydegree_OLS(ols_F_good_pred, ols_T_good_pred, filename="Figure3b.pdf")
plot.plot_mse_per_polydegree_OLS(ols_F, ols_T, filename="Figure4a.pdf")
plot.plot_r2_score_per_polydegree_OLS(ols_F, ols_T, filename="Figure4b.pdf")

# ----------------------------------------------- Figure 5a, 5b ----------------------------------------------

params = np.logspace(-8, 4, 13)
degree_F = 8    # by visual inspection, most overfitting
degree_T = 7    # by visual inspection, most overfitting

ridge_F = res.Results(
    RidgeRegression, 
    x_Franke, 
    y_Franke, 
    maxdegree, 
    params,
    multidim=multidim, 
    scale=scale, 
    with_std=with_std, 
    test_size=test_size
    )
ridge_F.train_and_predict_all_models()
ridge_F.calculate_MSE_across_degrees()
ridge_F.calculate_R2_across_degrees()

ridge_T = res.Results(
    RidgeRegression, 
    x_terrain, 
    y_terrain, 
    maxdegree, 
    params,
    multidim=multidim, 
    scale=scale, 
    with_std=with_std, 
    test_size=test_size
    )
ridge_T.train_and_predict_all_models()
ridge_T.calculate_MSE_across_degrees()
ridge_T.calculate_R2_across_degrees()

lasso_F = res.Results(
    LassoRegression, 
    x_Franke, 
    y_Franke,
    maxdegree,
    params, 
    multidim=multidim, 
    scale=scale, 
    with_std=with_std, 
    test_size=test_size
    )
lasso_F.train_and_predict_all_models()
lasso_F.calculate_MSE_across_degrees()
lasso_F.calculate_R2_across_degrees()

lasso_T = res.Results(
    LassoRegression, 
    x_terrain, 
    y_terrain, 
    maxdegree, 
    params,
    multidim=multidim, 
    scale=scale, 
    with_std=with_std, 
    test_size=test_size
    )
lasso_T.train_and_predict_all_models()
lasso_T.calculate_MSE_across_degrees()
lasso_T.calculate_R2_across_degrees()

instances_F = [ols_F, ridge_F, lasso_F]
instances_T = [ols_T, ridge_T, lasso_T]

plot.plot_mse_per_hyper_param_all_instances(
    res_instances_F=instances_F,
    res_instances_T=instances_T,
    degree_F=degree_F,
    degree_T=degree_T,
    filename="Figure5a.pdf"
)
plot.plot_r2_score_per_hyper_param_all_instances(
    res_instances_F=instances_F,
    res_instances_T=instances_T,
    degree_F=degree_F,
    degree_T=degree_T,
    filename="Figure5b.pdf"
)

# ----------------------------------------------- Figure 6 a,b,c,d -------------------------------------------

plot.plot_CV_table(ridge_F, "Franke Function", filename="Figure6a.pdf")
plot.plot_CV_table(ridge_T, "Terrain Data", filename="Figure6b.pdf")
plot.plot_CV_table(lasso_F, "Franke Function", filename="Figure6c.pdf")
plot.plot_CV_table(lasso_T, "Terrain Data", filename="Figure6d.pdf")


# ----------------------------------------------- Figure 7 a,b,c,d -------------------------------------------

param_F_1 = 0.001
param_F_2 = 1.0

param_T_1 = 1000.0
param_T_2 = 10000.0

plot.plot_mse_per_polydegree_all_instances(
    instances_F=instances_F, 
    instances_T=instances_T, 
    param_F=param_F_1, 
    param_T=param_T_1,
    filename="Figure7a.pdf"
)

plot.plot_r2_score_per_polydegree_all_instances(
    instances_F=instances_F, 
    instances_T=instances_T, 
    param_F=param_F_1, 
    param_T=param_T_1,
    filename="Figure7b.pdf"
)

plot.plot_mse_per_polydegree_all_instances(
    instances_F=instances_F, 
    instances_T=instances_T, 
    param_F=param_F_2, 
    param_T=param_T_2,
    filename="Figure7c.pdf"
)

plot.plot_r2_score_per_polydegree_all_instances(
    instances_F=instances_F, 
    instances_T=instances_T, 
    param_F=param_F_2, 
    param_T=param_T_2,
    filename="Figure7d.pdf"
)

# ----------------------------------------------- Figure 8 a,b,c ---------------------------------------------

plot.plot_optimal_coefficients(
    instances_F=instances_F, 
    instances_T=instances_T, 
    param_F=param_F_1, 
    param_T=param_T_1,
    maxdegree=5,
    filename="Figure8a.pdf"
)

plot.plot_optimal_coefficients(
    instances_F=instances_F, 
    instances_T=instances_T, 
    param_F=param_F_2,
    param_T=param_T_2,
    maxdegree=5,
    filename="figure8b.pdf"
)

plot.plot_optimal_coefficients(
    instances_F=instances_F, 
    instances_T=instances_T, 
    param_F=param_F_1, 
    param_T=param_T_1,
    maxdegree=maxdegree,
    filename="Figure8c.pdf"
)

plot.plot_optimal_coefficients(
    instances_F=instances_F, 
    instances_T=instances_T, 
    param_F=param_F_2,
    param_T=param_T_2,
    maxdegree=maxdegree,
    filename="figure8d.pdf"
)

# ----------------------------------------------- Table 1 ----------------------------------------------------

degree = 2

ols_F.print_correlation_matrix(degree)


# # ----------------------------------------------- Figure 9 a,b,c ---------------------------------------------

medium_n_F = int(n_F * 2)
medium_n_T = int(n_T * 2)

medium_terrain_step = int(terrain_step / 2)

x_F_medium_n, y_F_medium_n, = res.generate_data_Franke(medium_n_F, seed, multidim=multidim)
x_T_medium_n, y_T_medium_n = res.generate_data_terrain(medium_n_T, terrain_start, medium_terrain_step, terrain_file)


ols_F_high_n = res.Results(
    OrdinaryLeastSquares, 
    x_F_high_n, 
    y_F_high_n, 
    maxdegree, 
    multidim=multidim, 
    scale=scale, 
    with_std=with_std, 
    test_size=test_size
    )
ols_F_high_n.train_and_predict_all_models()
ols_F_high_n.calculate_MSE_across_degrees()
ols_F_high_n.calculate_R2_across_degrees()

ols_T_high_n = res.Results(
    OrdinaryLeastSquares, 
    x_T_high_n, 
    y_T_high_n, 
    maxdegree, 
    multidim=multidim, 
    scale=scale, 
    with_std=with_std, 
    test_size=test_size
    )
ols_T_high_n.train_and_predict_all_models()
ols_T_high_n.calculate_MSE_across_degrees()
ols_T_high_n.calculate_R2_across_degrees()


ols_F_medium_n = res.Results(
    OrdinaryLeastSquares, 
    x_F_medium_n, 
    y_F_medium_n, 
    maxdegree, 
    multidim=multidim, 
    scale=scale, 
    with_std=with_std, 
    test_size=test_size
    )
ols_F_medium_n.train_and_predict_all_models()
ols_F_medium_n.calculate_MSE_across_degrees()
ols_F_medium_n.calculate_R2_across_degrees()

ols_T_medium_n = res.Results(
    OrdinaryLeastSquares, 
    x_T_medium_n, 
    y_T_medium_n, 
    maxdegree, 
    multidim=multidim, 
    scale=scale, 
    with_std=with_std, 
    test_size=test_size
    )
ols_T_medium_n.train_and_predict_all_models()
ols_T_medium_n.calculate_MSE_across_degrees()
ols_T_medium_n.calculate_R2_across_degrees()

n_bootstrap = 100

plot.plot_error_bias_variance(ols_F, ols_T, n_bootstrap, filename="Figure9a.pdf")
plot.plot_error_bias_variance(ols_F_medium_n, ols_T_medium_n, n_bootstrap, filename="Figure9b.pdf")
plot.plot_error_bias_variance(ols_F_high_n, ols_T_high_n, n_bootstrap, "Figure9c.pdf")


# # ----------------------------------------------- Figure 10 --------------------------------------------------

kfolds = np.arange(5, 10 + 1)
kfold_degree = 4

kfold_params_F = [None, 1.0, 0.0001]
kfold_params_T = [None, 10e3, 10e4]

bootstrap_error_F, _, _ = ols_F.bootstrap_resampling(n_bootstrap)
bootstrap_error_T, _, _ = ols_T.bootstrap_resampling(n_bootstrap)

plot.plot_kfold_per_degree(
        instances_F=instances_F,
        instances_T=instances_T,
        kfolds=kfolds,
        degree=kfold_degree,
        params_F=kfold_params_F,
        params_T=kfold_params_T,
        bootstrap_error_F=bootstrap_error_F[kfold_degree - 1],
        bootstrap_error_T=bootstrap_error_T[kfold_degree - 1],
        filename="Figure10.pdf"
    )

# ----------------------------------------------- Figure 11 --------------------------------------------------

degree = 4

alpha_F_low_n = 0.001
alpha_F_high_n = 0.00001
alpha_T_low_n = 1000
alpha_T_high_n = 0.00001

LASSO_F_low_n = LassoRegression(maxdegree, alpha_F_low_n, multidim=multidim)
X_F_low_n = LASSO_F_low_n.create_design_matrix(x_Franke)
LASSO_F_low_n.fit(X_F_low_n, y_Franke)
X_F_low_n = LASSO_F_low_n.transform(X_F_low_n)
LASSO_F_low_n.train(X_F_low_n, y_Franke)
y_tilde_F_LASSO_low_n = LASSO_F_low_n.predict(X_F_low_n)

LASSO_F_high_n = LassoRegression(maxdegree, alpha_F_high_n, multidim=multidim)
X_F_high_n = LASSO_F_high_n.create_design_matrix(x_F_high_n)
LASSO_F_high_n.fit(X_F_high_n, y_F_high_n)
X_F_high_n = LASSO_F_high_n.transform(X_F_high_n)
LASSO_F_high_n.train(X_F_high_n, y_F_high_n)
y_tilde_F_LASSO_high_n = LASSO_F_high_n.predict(X_F_high_n)

LASSO_T_low_n = LassoRegression(degree, alpha_T_low_n, multidim=multidim)
X_T_low_n = LASSO_T_low_n.create_design_matrix(x_terrain)
LASSO_T_low_n.fit(X_T_low_n, y_terrain)
X_T_low_n = LASSO_T_low_n.transform(X_T_low_n)
LASSO_T_low_n.train(X_T_low_n, y_terrain)
y_tilde_T_LASSO_low_n = LASSO_T_low_n.predict(X_T_low_n)

LASSO_T_high_n = LassoRegression(degree, alpha_T_high_n, multidim=multidim)
X_T_high_n = LASSO_T_high_n.create_design_matrix(x_T_high_n)
LASSO_T_high_n.fit(X_T_high_n, y_F_high_n)
X_T_high_n = LASSO_T_high_n.transform(X_T_high_n)
LASSO_T_high_n.train(X_T_high_n, y_T_high_n)
y_tilde_T_LASSO_high_n = LASSO_T_high_n.predict(X_T_high_n)


plot.plot_true_vs_predicted_terrain(
    y_tilde_high_n=y_tilde_T_LASSO_high_n,
    y_tilde_low_n=y_tilde_T_LASSO_low_n,
    high_n=high_n_T,
    low_n=n_T,
    start=terrain_start,
    step_high_n=low_terrain_step,
    step_low_n=terrain_step,
    terrain_file=terrain_file,
    filename="Figure11a.pdf"
)

maxdegree = 6

OLS_F_low_n = OrdinaryLeastSquares(maxdegree, multidim=multidim)
X_F_low_n = OLS_F_low_n.create_design_matrix(x_Franke)
OLS_F_low_n.fit(X_F_low_n, y_Franke)
X_F_low_n = OLS_F_low_n.transform(X_F_low_n)
OLS_F_low_n.train(X_F_low_n, y_Franke)
y_tilde_F_OLS_low_n = OLS_F_low_n.predict(X_F_low_n)

OLS_F_high_n = OrdinaryLeastSquares(maxdegree, multidim=multidim)
X_F_high_n = OLS_F_high_n.create_design_matrix(x_F_high_n)
OLS_F_high_n.fit(X_F_high_n, y_F_high_n)
X_F_high_n = OLS_F_high_n.transform(X_F_high_n)
OLS_F_high_n.train(X_F_high_n, y_F_high_n)
y_tilde_F_OLS_high_n = OLS_F_high_n.predict(X_F_high_n)

OLS_T_low_n = OrdinaryLeastSquares(maxdegree, multidim=multidim)
X_T_low_n = OLS_T_low_n.create_design_matrix(x_terrain)
OLS_T_low_n.fit(X_T_low_n, y_terrain)
X_T_low_n = OLS_T_low_n.transform(X_T_low_n)
OLS_T_low_n.train(X_T_low_n, y_terrain)
y_tilde_T_OLS_low_n = OLS_T_low_n.predict(X_T_low_n)

OLS_T_high_n = OrdinaryLeastSquares(maxdegree, multidim=multidim)
X_T_high_n = OLS_T_high_n.create_design_matrix(x_T_high_n)
OLS_T_high_n.fit(X_T_high_n, y_F_high_n)
X_T_high_n = OLS_T_high_n.transform(X_T_high_n)
OLS_T_high_n.train(X_T_high_n, y_T_high_n)
y_tilde_T_OLS_high_n = OLS_T_high_n.predict(X_T_high_n)


lambda_F_low_n = 0.00001
lambda_F_high_n = 0.00001
lambda_T_low_n = 0.00001
lambda_T_high_n = 0.00001

RIDGE_F_low_n = RidgeRegression(maxdegree, lambda_F_low_n, multidim=multidim)
X_F_low_n = RIDGE_F_low_n.create_design_matrix(x_Franke)
RIDGE_F_low_n.fit(X_F_low_n, y_Franke)
X_F_low_n = RIDGE_F_low_n.transform(X_F_low_n)
RIDGE_F_low_n.train(X_F_low_n, y_Franke)
y_tilde_F_RIDGE_low_n = RIDGE_F_low_n.predict(X_F_low_n)

RIDGE_F_high_n = RidgeRegression(maxdegree, lambda_F_high_n, multidim=multidim)
X_F_high_n = RIDGE_F_high_n.create_design_matrix(x_F_high_n)
RIDGE_F_high_n.fit(X_F_high_n, y_F_high_n)
X_F_high_n = RIDGE_F_high_n.transform(X_F_high_n)
RIDGE_F_high_n.train(X_F_high_n, y_F_high_n)
y_tilde_F_RIDGE_high_n = RIDGE_F_high_n.predict(X_F_high_n)

RIDGE_T_low_n = RidgeRegression(maxdegree, lambda_T_low_n, multidim=multidim)
X_T_low_n = RIDGE_T_low_n.create_design_matrix(x_terrain)
RIDGE_T_low_n.fit(X_T_low_n, y_terrain)
X_T_low_n = RIDGE_T_low_n.transform(X_T_low_n)
RIDGE_T_low_n.train(X_T_low_n, y_terrain)
y_tilde_T_RIDGE_low_n = RIDGE_T_low_n.predict(X_T_low_n)

RIDGE_T_high_n = RidgeRegression(maxdegree, lambda_F_high_n, multidim=multidim)
X_T_high_n = RIDGE_T_high_n.create_design_matrix(x_T_high_n)
RIDGE_T_high_n.fit(X_T_high_n, y_F_high_n)
X_T_high_n = RIDGE_T_high_n.transform(X_T_high_n)
RIDGE_T_high_n.train(X_T_high_n, y_T_high_n)
y_tilde_T_RIDGE_high_n = RIDGE_T_high_n.predict(X_T_high_n)



alpha_F_low_n = 0.00001
alpha_F_high_n = 0.00001
alpha_T_low_n = 0.00001
alpha_T_high_n = 0.00001

LASSO_F_low_n = LassoRegression(maxdegree, alpha_F_low_n, multidim=multidim)
X_F_low_n = LASSO_F_low_n.create_design_matrix(x_Franke)
LASSO_F_low_n.fit(X_F_low_n, y_Franke)
X_F_low_n = LASSO_F_low_n.transform(X_F_low_n)
LASSO_F_low_n.train(X_F_low_n, y_Franke)
y_tilde_F_LASSO_low_n = LASSO_F_low_n.predict(X_F_low_n)

LASSO_F_high_n = LassoRegression(maxdegree, alpha_F_high_n, multidim=multidim)
X_F_high_n = LASSO_F_high_n.create_design_matrix(x_F_high_n)
LASSO_F_high_n.fit(X_F_high_n, y_F_high_n)
X_F_high_n = LASSO_F_high_n.transform(X_F_high_n)
LASSO_F_high_n.train(X_F_high_n, y_F_high_n)
y_tilde_F_LASSO_high_n = LASSO_F_high_n.predict(X_F_high_n)

LASSO_T_low_n = LassoRegression(maxdegree, alpha_T_low_n, multidim=multidim)
X_T_low_n = LASSO_T_low_n.create_design_matrix(x_terrain)
LASSO_T_low_n.fit(X_T_low_n, y_terrain)
X_T_low_n = LASSO_T_low_n.transform(X_T_low_n)
LASSO_T_low_n.train(X_T_low_n, y_terrain)
y_tilde_T_LASSO_low_n = LASSO_T_low_n.predict(X_T_low_n)

LASSO_T_high_n = LassoRegression(maxdegree, alpha_F_high_n, multidim=multidim)
X_T_high_n = LASSO_T_high_n.create_design_matrix(x_T_high_n)
LASSO_T_high_n.fit(X_T_high_n, y_F_high_n)
X_T_high_n = LASSO_T_high_n.transform(X_T_high_n)
LASSO_T_high_n.train(X_T_high_n, y_T_high_n)
y_tilde_T_LASSO_high_n = LASSO_T_high_n.predict(X_T_high_n)


plot.plot_true_vs_predicted_Franke(
    y_tilde_high_n=y_tilde_F_OLS_high_n,
    y_tilde_low_n=y_tilde_F_OLS_low_n,
    high_n=high_n_F,
    low_n=n_F,
    filename="Figure11b.pdf"
)

plot.plot_true_vs_predicted_terrain(
    y_tilde_high_n=y_tilde_T_OLS_high_n,
    y_tilde_low_n=y_tilde_T_OLS_low_n,
    high_n=high_n_T,
    low_n=n_T,
    start=terrain_start,
    step_high_n=low_terrain_step,
    step_low_n=terrain_step,
    terrain_file=terrain_file,
    filename="Figure11c.pdf"
)


plot.plot_true_vs_predicted_Franke(
    y_tilde_high_n=y_tilde_F_RIDGE_high_n,
    y_tilde_low_n=y_tilde_F_RIDGE_low_n,
    high_n=high_n_F,
    low_n=n_F,
    filename="Figure11d.pdf"
)

plot.plot_true_vs_predicted_terrain(
    y_tilde_high_n=y_tilde_T_RIDGE_high_n,
    y_tilde_low_n=y_tilde_T_RIDGE_low_n,
    high_n=high_n_T,
    low_n=n_T,
    start=terrain_start,
    step_high_n=low_terrain_step,
    step_low_n=terrain_step,
    terrain_file=terrain_file,
    filename="Figure11e.pdf"
)


plot.plot_true_vs_predicted_Franke(
    y_tilde_high_n=y_tilde_F_LASSO_high_n,
    y_tilde_low_n=y_tilde_F_LASSO_low_n,
    high_n=high_n_F,
    low_n=n_F,
    filename="Figure11f.pdf"
)

plot.plot_true_vs_predicted_terrain(
    y_tilde_high_n=y_tilde_T_LASSO_high_n,
    y_tilde_low_n=y_tilde_T_LASSO_low_n,
    high_n=high_n_T,
    low_n=n_T,
    start=terrain_start,
    step_high_n=low_terrain_step,
    step_low_n=terrain_step,
    terrain_file=terrain_file,
    filename="Figure11g.pdf"
)

maxdegree = 2

lambda_F_low_n = 1.0
lambda_T_low_n = 10000

RIDGE_F_low_n = RidgeRegression(maxdegree, lambda_F_low_n, multidim=multidim)
X_F_low_n = RIDGE_F_low_n.create_design_matrix(x_Franke)
RIDGE_F_low_n.fit(X_F_low_n, y_Franke)
X_F_low_n = RIDGE_F_low_n.transform(X_F_low_n)
RIDGE_F_low_n.train(X_F_low_n, y_Franke)
y_tilde_F_RIDGE_low_n = RIDGE_F_low_n.predict(X_F_low_n)


maxdegree = 1

RIDGE_T_low_n = RidgeRegression(maxdegree, lambda_T_low_n, multidim=multidim)
X_T_low_n = RIDGE_T_low_n.create_design_matrix(x_terrain)
RIDGE_T_low_n.fit(X_T_low_n, y_terrain)
X_T_low_n = RIDGE_T_low_n.transform(X_T_low_n)
RIDGE_T_low_n.train(X_T_low_n, y_terrain)
y_tilde_T_RIDGE_low_n = RIDGE_T_low_n.predict(X_T_low_n)


plot.plot_true_vs_predicted_Franke_low_n(
    y_tilde_low_n=y_tilde_F_RIDGE_low_n,
    low_n=n_F,
    filename="Figure11h.pdf"
)

plot.plot_true_vs_predicted_terrain_low_n(
    y_tilde_low_n=y_tilde_T_RIDGE_low_n,
    low_n=n_T,
    start=terrain_start,
    step_low_n=terrain_step,
    terrain_file=terrain_file,
    filename="Figure11i.pdf"
)
