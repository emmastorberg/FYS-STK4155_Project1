
import numpy as np

from LinearRegression import OrdinaryLeastSquares, RidgeRegression, LassoRegression
import results as res
import plotroutines as plot

n = 50
seed = 55555
maxdegree = 5
multidim = True
scale = True
with_std = True
test_size = 0.2

terrain_file = "datasets/SRTM_data_Norway_1.tif"
terrain_start = 700

x_Franke, y_Franke, = res.generate_data(n, seed, multidim=multidim)
x_terrain, y_terrain = res.generate_terrain_data(n, terrain_start, terrain_file)

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

plot.plot_mse_polydegree_OLS(ols_F, ols_T)



