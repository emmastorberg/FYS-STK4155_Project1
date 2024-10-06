import pytest

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.preprocessing import scale, PolynomialFeatures, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold


import LinearRegression as linreg
from results import Results, generate_data_Franke, generate_data_terrain 
import plotroutines


@pytest.mark.parametrize(
        "n, degree, seed", [(10, 2, 8), (50, 5, 103), (100, 5, 45)]
)
def test_multivariate_design_matrix(n, degree, seed):
    """
    Test construction of multivariate design matrix. Compare to scikit-learn.
    """
    X, Y = generate_data_Franke(n, seed, multidim=True)
    X1, X2 = X

    OLS = linreg.BaseModel(degree, multidim=True)
    X = OLS.create_design_matrix((X1, X2))

    # All combinations of (x, y) pairs
    data = np.hstack((X1, X2))

    poly = PolynomialFeatures(degree, include_bias=False)
    X_sklearn = poly.fit_transform(data)

    assert np.allclose(X, X_sklearn)


@pytest.mark.parametrize(
        "n, degree, seed", [(10, 2, 8), (100, 5, 103), (100, 10, 45)]
)
def test_univariate_design_matrix(n, degree, seed):
    """
    Test construction of univariate design matrix. Compare to scikit-learn.
    """
    x, y = generate_data_Franke(n, seed, multidim=False)
    y = np.zeros(n)

    OLS = linreg.BaseModel(degree, multidim=False)
    X = OLS.create_design_matrix(x)

    poly = PolynomialFeatures(degree, include_bias=False)
    X_sklearn = poly.fit_transform(x)

    assert np.allclose(X, X_sklearn)


@pytest.mark.parametrize(
        "n, degree, seed, multidim", [(10, 2, 8, True), (100, 5, 103, False), (100, 10, 45, True)]
)
def test_fit_scaler_transform(n, degree, seed, multidim):
    x, y = generate_data_Franke(n, seed, multidim=multidim)

    OLS = linreg.BaseModel(degree, multidim=multidim)
    X = OLS.create_design_matrix(x)
    X_train, X_test, y_train, y_test = OLS.split_test_train(X, y)
    OLS.fit(X_train, y_train)
    X_train_scaled = OLS.transform(X_train)
    X_test_scaled = OLS.transform(X_test)

    model = StandardScaler()
    X_train_sklearn = model.fit_transform(X_train)
    X_test_sklearn = model.transform(X_test)

    assert np.allclose(X_train_scaled, X_train_sklearn)
    assert np.allclose(X_test_scaled, X_test_sklearn)


@pytest.mark.parametrize(
        "y_true, y_pred", [
                (np.linspace(1, 10, 1001), np.linspace(1, 10, 1001)), 
                ([1, 5, 20, 7], [2, 4, 10, 30]),
                ([0, 1], [0.5, 0.5]),
            ]
)
def test_MSE(y_true, y_pred):
    """
    Test static method 'calculate_MSE'. Compare to scikit-learn.
    """
    expected = mean_squared_error(y_true, y_pred)
    computed = Results.mean_squared_error(y_true, y_pred)
    assert np.allclose(expected, computed)


@pytest.mark.parametrize(
        "y_true, y_pred", [
                (np.linspace(1, 10, 1001), np.linspace(1, 10, 1001)), 
                ([1, 5, 20, 7], [2, 4, 10, 30]),
                ([0, 0.5, 1], [0.5, 0.5, 0.5]),
            ]
)
def test_R2_score(y_true, y_pred):
    """
    Test static method 'calculate_R2_score'. Compare to scikit-learn.
    """
    expected = r2_score(y_true, y_pred)
    computed = Results.r2_score(y_true, y_pred)
    assert abs(expected - computed) < 1e-10


@pytest.mark.parametrize(
        "n, degree, func", [
            (5, 1, lambda X1, X2: X1**2 + X2**2), 
            (100, 3, lambda X1, X2: X1 + X2), 
            (50, 5, lambda X1, X2: 4 * np.sin(X1) + X2**4),
        ]
)
def test_compute_optimal_beta_OLS_multivariate(n, degree, func):
    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(0, 1, n)
    X1, X2 = np.meshgrid(x1, x2)
    Y = func(X1, X2)
    X1 = X1.flatten().reshape(-1, 1)
    X2 = X2.flatten().reshape(-1, 1)
    Y = Y.flatten().reshape(-1, 1)
    data = np.hstack((X1, X2))

    OLS = linreg.OrdinaryLeastSquares(degree, multidim=True)
    X = OLS.create_design_matrix((X1, X2))
    beta = OLS.train(X, Y)

    poly = PolynomialFeatures(degree, include_bias=False)
    design_matrix_sklearn = poly.fit_transform(data)
    model = LinearRegression(fit_intercept=False)
    model.fit(design_matrix_sklearn, Y)
    beta_sklearn = model.coef_.reshape(-1, 1)
    assert np.allclose(beta, beta_sklearn)


@pytest.mark.parametrize(
        "n, degree, func", [
            (5, 1, lambda x: x**2), 
            (100, 3, lambda x: 2 * x**3 + x**2), 
            (50, 5, lambda x: 4 * np.sin(8*x)),
        ]
)
def test_compute_optimal_beta_OLS_univariate(n, degree, func):
    x = np.linspace(0, 1, n).reshape(-1, 1)
    y = func(x)

    OLS = linreg.OrdinaryLeastSquares(degree, multidim=False)
    X = OLS.create_design_matrix(x)
    beta = OLS.train(X, y)

    poly = PolynomialFeatures(degree, include_bias=False)
    design_matrix_sklearn = poly.fit_transform(x)
    model = LinearRegression(fit_intercept=False)
    model.fit(design_matrix_sklearn, y)
    beta_sklearn = model.coef_.reshape(-1, 1)
    assert np.allclose(beta, beta_sklearn)


@pytest.mark.parametrize(
        "n, degree, func, lmbda", [
            (5, 1, lambda X1, X2: X1**2 + X2**2, 0.001), 
            (100, 3, lambda X1, X2: X1 + X2, 1), 
            (50, 5, lambda X1, X2: 4 * np.sin(X1) + X2**4, 0.5),
        ]
)
def test_compute_optimal_beta_Ridge_multivariate(n, degree, func, lmbda):
    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(0, 1, n)
    X1, X2 = np.meshgrid(x1, x2)
    Y = func(X1, X2)
    X1 = X1.flatten().reshape(-1, 1)
    X2 = X2.flatten().reshape(-1, 1)
    Y = Y.flatten().reshape(-1, 1)
    data = np.hstack((X1, X2))

    OLS = linreg.RidgeRegression(degree, lmbda, multidim=True)
    X = OLS.create_design_matrix((X1, X2))
    beta = OLS.train(X, Y)

    poly = PolynomialFeatures(degree, include_bias=False)
    design_matrix_sklearn = poly.fit_transform(data)
    model = Ridge(alpha=lmbda, fit_intercept=False)
    model.fit(design_matrix_sklearn, Y)
    beta_sklearn = model.coef_.reshape(-1, 1)
    assert np.allclose(beta, beta_sklearn)


@pytest.mark.parametrize(
        "n, degree, func, lmbda", [
            (5, 1, lambda x: x**2, 0.001), 
            (100, 3, lambda x: 2 * x**3 + x**2, 1), 
            (50, 5, lambda x: 4 * np.sin(8*x), 0.0000001),
        ]
)
def test_compute_optimal_beta_Rdige_univariate(n, degree, func, lmbda):
    x = np.linspace(0, 1, n).reshape(-1, 1)
    y = func(x)

    OLS = linreg.RidgeRegression(degree, lmbda, multidim=False)
    X = OLS.create_design_matrix(x)
    beta = OLS.train(X, y)

    poly = PolynomialFeatures(degree, include_bias=False)
    design_matrix_sklearn = poly.fit_transform(x)
    model = Ridge(alpha=lmbda, fit_intercept=False)
    model.fit(design_matrix_sklearn, y)
    beta_sklearn = model.coef_.reshape(-1, 1)

    assert np.allclose(beta, beta_sklearn)


def test_kfold_CV():
    maxdegree = 6
    nsamples = 100
    nlambdas = 500
    lambdas = np.logspace(-3, 5, nlambdas)
    k = 5

    x = np.random.randn(nsamples).reshape(-1, 1)
    y = 3*x**2 + np.random.randn(nsamples).reshape(-1,1)
    y = y.ravel()

    estimated_mse_sklearn = np.empty(len(lambdas))
    estimated_mse = np.empty(len(lambdas))

    ridge = Results(
        linreg.RidgeRegression, 
        x,
        y,
        maxdegree, 
        lambdas, 
        multidim=False,
        scale=False,
        )
    ridge.train_and_predict_all_models()
    
    for i, param in enumerate(lambdas):        
        model = make_pipeline(PolynomialFeatures(degree=maxdegree), StandardScaler(), Ridge(alpha=param, fit_intercept=False))
        estimated_mse_folds = cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=k)
        estimated_mse_sklearn[i] = np.mean(-estimated_mse_folds)
        estimated_mse[i] = ridge.kfold_CV(k, maxdegree, param)

    assert np.allclose(estimated_mse_sklearn, estimated_mse_sklearn)


def test_predict_ridge():
    n = 15
    start = 0
    step = 1
    filename="datasets/SRTM_data_Norway_1.tif"
    (x1, x2), y = generate_data_terrain(n, start, step, filename)

    degree = 4
    param = 0.1

    ridge = linreg.RidgeRegression(degree, param, multidim=True)
    X = ridge.create_design_matrix((x1, x2))
    ridge.fit(X, y, with_std=True)
    X_my = ridge.transform(X)
    ridge.train(X_my, y)
    y_tilde = ridge.predict(X_my)

    scaler = StandardScaler(with_std=True)
    X_sklearn = scaler.fit_transform(X)
    ridge_sklearn = Ridge(alpha=param)
    ridge_sklearn.fit(X_sklearn, y)
    y_tilde_sklearn = ridge_sklearn.predict(X_sklearn)

    assert np.allclose(y_tilde, y_tilde_sklearn)


if __name__ == "__main__":
    compare_kfold_to_sklearn()
    

