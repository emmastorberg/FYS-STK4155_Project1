from collections.abc import Iterable

import numpy as np
from sklearn.model_selection import KFold
from imageio.v2 import imread
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error

from LinearRegression import BaseModel, OrdinaryLeastSquares, RidgeRegression, LassoRegression


class Results:
    def __init__(self, model: BaseModel, x, y, maxdegree, params=None, multidim=True, scale=True, with_std=True, test_size=0.2):
        self.model = model
        self.x = x
        self.y = y
        self.maxdegree = maxdegree
        if not isinstance(params, Iterable):
            params = [params]
        self.params = params
        self.multidim = multidim
        self.scale = scale
        self.with_std = with_std
        self.test_size = test_size

        self.name = None
        self.param_label = None
        self.degrees = range(1, maxdegree + 1)
        self.X = {}
        self.X_train = {}
        self.X_test = {}
        self.y_train = {}
        self.y_test = {}
        self.beta_hat = {}
        self.mse_train = {}
        self.mse_test = {}
        self.r2_score_train = {}
        self.r2_score_test = {}
        self.y_tilde_train = {}
        self.y_tilde_test = {}

    def train_and_predict_all_models(self):
        """
        Train, predict and store values across all degrees and hyper parameters.
        """
        for param in self.params:
            self.beta_hat[param] = {}
            self.y_tilde_train[param] = {}
            self.y_tilde_test[param] = {}
            for degree in self.degrees:
                linreg = self.model(degree, param, multidim=self.multidim)
                X = linreg.create_design_matrix(self.x)
                X_train, X_test, y_train, y_test = linreg.split_test_train(
                    X, self.y, test_size=self.test_size
                )
                if self.scale:
                    linreg.fit(X_train, y_train, with_std=self.with_std)
                    X_train = linreg.transform(X_train)
                    X_test = linreg.transform(X_test)
                beta_hat = linreg.train(X_train, y_train)
                y_tilde_train = linreg.predict(X_train)
                y_tilde_test = linreg.predict(X_test)

                self.X[degree] = X
                self.X_train[degree] = X_train
                self.X_test[degree] = X_test
                self.y_train[degree] = y_train
                self.y_test[degree] = y_test
                self.beta_hat[param][degree] = beta_hat
                self.y_tilde_train[param][degree] = y_tilde_train
                self.y_tilde_test[param][degree] = y_tilde_test
        self.param_label = linreg.param_label
        self.name = linreg.name

    @staticmethod
    def mean_squared_error(y: np.ndarray, y_tilde: np.ndarray) -> float:
        """Calculates mean squared error of predicted y_tilde compared to
        some known output values y.

        Args:
            y (np.ndarray): known y values
            y_tilde (np.ndarray): y values as predicted by model

        Returns:
            float: calculated mean squared error
        """
        try:
            y = np.array(y)
            y_tilde = np.array(y_tilde)
        except:
            raise TypeError("input must be iterable")
        return np.sum((y - y_tilde)**2) / len(y)

    @staticmethod
    def r2_score(y: np.ndarray, y_tilde: np.ndarray) -> float:
        """Calculates R2 score of predicted y_tilde comapred some known output 
        values y.

        Args:
            y (np.ndarray): known y values
            y_tilde (np.ndarray): y values as predicted by model

        Returns:
            float: calculated R2 score
        """
        try:
            y = np.array(y)
            y_tilde = np.array(y_tilde)
        except:
            raise TypeError("input must be iterable")
        R2_score = 1 - (np.sum((y - y_tilde) ** 2) / np.sum((y - np.mean(y))**2))
        return R2_score

    def calculate_MSE_across_degrees(self) -> None:
        """Fills dictionaries of MSE from training and 
        testing data with lists containing of MSE values. 
        Keys are the different parameter values, and each 
        parameter has MSE calculated for all degrees up
        to self.maxdegree.
        """
        for param in self.params:
            self.mse_train[param] = np.zeros(self.maxdegree)
            self.mse_test[param] = np.zeros(self.maxdegree)
            for degree in self.degrees:
                mse_train = self.mean_squared_error(
                    self.y_train[degree], self.y_tilde_train[param][degree]
                )
                mse_test = self.mean_squared_error(
                    self.y_test[degree], self.y_tilde_test[param][degree]
                )
                self.mse_train[param][degree - 1] = mse_train
                self.mse_test[param][degree - 1] = mse_test

    def calculate_R2_across_degrees(self) -> None:
        """Fills dictionaries of R2 scores from training 
        and testing data with lists containing of R2 scores. 
        Keys are the different parameter values, and each 
        parameter has R2 calculated for all degrees up
        to self.maxdegree.
        """
        for param in self.params:
            self.r2_score_train[param] = np.zeros(self.maxdegree)
            self.r2_score_test[param] = np.zeros(self.maxdegree)
            for degree in range(1, self.maxdegree + 1):
                r2_score_train = self.r2_score(
                    self.y_train[degree], self.y_tilde_train[param][degree]
                )
                r2_score_test = self.r2_score(
                    self.y_test[degree], self.y_tilde_test[param][degree]
                )
                self.r2_score_train[param][degree - 1] = r2_score_train
                self.r2_score_test[param][degree - 1] = r2_score_test

    def kfold_CV(self, k:  int, degree: int, param: float) -> float:
        """Performs K-Fold cross validation resampling.

        Args:
            k (int): number of folds
            degree (int): degree of model to look at
            param (float): parameter to be given to model

        Returns:
            float: mean squared error after cross validation
        """
        X, y = self.X[degree], self.y
        kfold = KFold(n_splits=k)
        scores_kfold = np.zeros(k)
        i = 0

        for train_inds, test_inds in kfold.split(X):
            X_train = X[train_inds]
            y_train = y[train_inds]

            X_test = X[test_inds]
            y_test = y[test_inds]

            linreg = self.model(degree, multidim=self.multidim, param=param)

            if self.scale:
                linreg.fit(X_test, y_test, with_std=self.with_std)
                X_train = linreg.transform(X_train)
                X_test = linreg.transform(X_test)

            linreg.train(X_train, y_train)
            y_pred_test = linreg.predict(X_test)

            MSE_test = self.mean_squared_error(y_test, y_pred_test)
            scores_kfold[i] = MSE_test
            i += 1
        return np.mean(scores_kfold)

    def grid_search(self) -> pd.DataFrame:
        """
        Perform a grid search to evaluate the model's performance 
        over a range of hyperparameters and polynomial degrees.

        This method uses k-fold cross-validation to compute the Mean Squared Error (MSE) 
        for each combination of hyperparameter and polynomial degree specified in the input.

        Args:
        params (list | None): A list of hyperparameter values to be evaluated during the grid search. 
        If `params` is None, the parameter values provided to the constructor is used.

        Returns:
        (pd.DataFrame): A DataFrame containing the cross-validated MSE for each combination of 
        hyperparameter and polynomial degree.

        Notes:
        The DataFrame's columns correspond to the hyperparameter values, 
        while the index corresponds to the polynomial degrees.
        """
        cv_grid = {}
        for param in self.params:
            cv_grid[param] = {}
            for degree in self.degrees:
                mse_cv = self.kfold_CV(5, degree, param)
                cv_grid[param][degree] = mse_cv
        df = pd.DataFrame(cv_grid)
        return df
    
    def bootstrap_resampling(self, num_bootstraps: int = 100, param=None) -> tuple[np.ndarray]:
        """Performs bias-variance analysis of dataset using bootstrap resampling.

        Args:
            num_bootstraps (int, optional): Number of bootstrap resamplings. Defaults to 100.

        Returns:
            tuple[np.ndarray]: a tuple with arrays for error, bias and variance.
        """
        error = np.zeros(self.maxdegree)
        bias = np.zeros(self.maxdegree)
        variance = np.zeros(self.maxdegree)

        for degree in self.degrees:

            X = self.X[degree]
            linreg = self.model(degree, self.multidim, param)
            X_train, X_test, y_train, y_test = linreg.split_test_train(X, self.y, test_size=self.test_size)

            if self.scale:
                linreg.fit(X_train, y_train)
                X_train = linreg.transform(X_train)
                X_test = linreg.transform(X_test)

            y_pred = np.zeros((y_test.shape[0], num_bootstraps))

            for i in range(num_bootstraps):
                X_, y_ = resample(X_train, y_train)
                linreg.train(X_, y_)
                y_tilde = linreg.predict(X_test)
                y_pred[:, i] = y_tilde.ravel()

            error[degree - 1] = np.mean(np.mean((y_test - y_pred)**2, axis=1, keepdims=True))
            bias[degree - 1] = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True))**2)
            variance[degree - 1] = np.mean(np.var(y_pred, axis=1, keepdims=True))

        return error, bias, variance
        
    def print_correlation_matrix(self, degree: int | None = None) -> None:
        """
        Print the correlation matrix for the specified polynomial degree.

        Args:
            polynomial_degree (int, optional): The polynomial degree for which to print the correlation matrix.

        Returns:
            None
        """
        if degree is None:
            degree = self.maxdegree
        X = self.X[degree]
        Xpd = pd.DataFrame(X)
        Xpd = Xpd - Xpd.mean()
        print(Xpd)
        correlation_matrix = Xpd.corr()
        formatted_matrix = correlation_matrix.round(2)
        print(formatted_matrix)


def Franke_function(x: np.ndarray, y: np.ndarray, noise: bool = True) -> np.ndarray:
    """Generates output data from Franke function, with optional noise.

    Args:
        x (np.ndarray): input values
        y (np.ndarray): input values
        noise (bool, optional): Boolean deciding whether or not to make noisy data. Defaults to True.

    Returns:
        np.ndarray: Output after input data is given to Franke function, and noise is possibly applied.
    """
    term1 = 3 / 4 * np.exp(-((9 * x - 2) ** 2) / 4 - ((9 * y - 2) ** 2) / 4)
    term2 = 3 / 4 * np.exp(-((9 * x + 1) ** 2) / 49 - (9 * y + 1) / 10)
    term3 = 1 / 2 * np.exp(-((9 * x - 7) ** 2) / 4 - ((9 * y - 3) ** 2) / 4)
    term4 = -1 / 5 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)

    Franke = term1 + term2 + term3 + term4

    if noise:
        state = np.random.get_state()
        Franke += np.random.normal(0, 0.1, x.shape)
        np.random.set_state(state)

    return Franke


def generate_data_Franke(n: int, seed: int | None = None, multidim: bool = False, noise: bool = True) -> tuple[np.ndarray]:
    """Generates noisy data to be given to our model. Can give multivariate or univariate data.

    Args:
        n (int): number of data points
        seed (int or None): set seed for consistency with random noise
        multidim (bool, optional): Whether or not to make the data multivariate. Defaults to False.

    Returns:
        tuple[np.ndarray]: Input and output data in a tuple. If multivariate, input data is itself 
        a tuple of various inputs X1 and X2.
    """
    np.random.seed(seed)
    if multidim:
        x1 = np.linspace(0, 1, n)
        x2 = np.linspace(0, 1, n)
        X1, X2 = np.meshgrid(x1, x2)
        Y = Franke_function(X1, X2, noise)

        x1 = X1.flatten().reshape(-1, 1)
        x2 = X2.flatten().reshape(-1, 1)
        y = Y.flatten().reshape(-1, 1)

        return (x1, x2), y

    else:
        x = np.linspace(-3, 3, n).reshape(-1, 1)
        y = (
            np.exp(-(x**2))
            + 1.5 * np.exp(-((x - 2) ** 2))
            + np.random.normal(0, 0.1, x.shape)
        )
        return x, y
    
    
def generate_data_terrain(n, start, step=1, filename="datasets/SRTM_data_Norway_1.tif"):
    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(0, 1, n)
    X1, X2 = np.meshgrid(x1, x2)

    terrain = imread(filename)
    Y  = terrain[start : start + (n * step) : step, start : start+ (n * step) : step]

    x1 = X1.flatten().reshape(-1, 1)
    x2 = X2.flatten().reshape(-1, 1)
    y = Y.flatten().reshape(-1, 1)

    return (x1, x2), y
