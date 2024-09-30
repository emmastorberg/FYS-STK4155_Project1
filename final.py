from collections.abc import Iterable

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.ticker import MultipleLocator
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample
import pandas as pd


class Model:
# ------------------------------------------------------------- INTERNAL METHODS ---------------------------------------------------------
    def __init__(
        self,
        X: np.ndarray | tuple[np.ndarray],
        Y: np.ndarray,
        maxdegree: int,
        params: list[int | None],
        test_size: float,
        scale: bool,
        multidim: bool,
    ) -> None:
        """Initializes an instance of the superclass for all models.

        Args:
            X (np.ndarray | tuple[np.ndarray]): Input data
            Y (np.ndarray): Output data
            maxdegree (int): Maximum degree of polynomial model to create. The class will create models up to and including this degree.
            params (list | int): list or integer with parameter to be given to cost function of model.
            test_size (float, optional): Ratio of data to be set aside for testing. Defaults to 0.2.
            scale (bool, optional): Whether or not to scale the data. Defaults to True.
            multidim (bool, optional): Whether or not the dataset has multivariate inputs. Defaults to False.
        """
        # May have X, Y, Z when all is multivairate
        self.X = X
        self.Y = Y.flatten().reshape(-1, 1)
        self.maxdegree = maxdegree
        self.degrees = np.arange(1, maxdegree + 1)
        self.params = params
        self.test_size = test_size
        self.scale = scale
        self.multidim = multidim
        self.design_matrices = self._generate_design_matrices()
        self.X_train, self.X_test, self.y_train, self.y_test = (
            self._split_train_test_data()
        )

        self.beta_hat = {}
        self.y_tilde_train = {}
        self.y_tilde_test = {}
        self.MSE_train = {}
        self.MSE_test = {}
        self.R2_score_train = {}
        self.R2_score_test = {}
        self.param_label = None

    def _generate_design_matrices(self) -> dict:
        """Generates dictionary of design matrices with 
        degree as keys.

        Returns:
            dict: dictionary containing design matrix for 
            each degree up to self.maxdegree with degree 
            integer as key
        """
        design_matrices = {}
        for degree in self.degrees:
            design_matrices[degree] = self._create_design_matrix(degree=degree)
        return design_matrices

    def _create_design_matrix(self, degree: int) -> np.ndarray:
        """Creates design matrix of a specified degree. Makes 
        multivariate design matrix if self.multidim == True.

        Args:
            degree (int): degree of design matrix to be generated

        Returns:
            np.ndarray: design matrix for relevant degree, may 
            be univariate or multivariate depending on self.multidim
        """
        if self.multidim:
            # multivariate
            X1, X2 = self.X
            X1 = X1.flatten()
            X2 = X2.flatten()

            n = len(X1)
            p = int(
                (degree + 1) * (degree + 2) / 2 - 1
            )  # Number of terms in the polynomial expansion

            X = np.zeros((n, p))

            for i in range(degree):
                base_index = int(
                    (i + 1) * (i + 2) / 2 - 1
                )  # Calculate base index for the current degree

                for j in range(i + 2):
                    # Fill the design matrix with x and y raised to appropriate powers
                    X[:, base_index + j] = X1 ** (i + 1 - j) * X2 ** (j)
            return X

        else:
            # univariate
            X = np.zeros((len(self.X), degree))

            for i in range(degree):
                X[:, i] = self.X[:, 0] ** (i + 1)
            return X

    def _split_train_test_data(self) -> tuple[dict]:
        """Generates dictionaries containing split training and 
        test data for all degrees up to self.maxdegree. Scaling
        is applied if self.scale == True.

        Returns:
            tuple[dict]: dictionaries containing training and testing 
            data with degree integer as keys
        """
        X_train = {}
        X_test = {}
        y_train = {}
        y_test = {}
        for d, X in self.design_matrices.items():
            X_train[d], X_test[d], y_train[d], y_test[d] = train_test_split(
                X, self.Y, test_size=self.test_size
            )
            if self.scale:
                X_train[d] = scale(X_train[d], with_std=False)
                X_test[d] = scale(X_test[d], with_std=False)
        return X_train, X_test, y_train, y_test

    def _calculate_MSE(self, y: np.ndarray, y_tilde: np.ndarray) -> float:
        """Calculates mean squared error of predicted y_tilde compared to
        some known output values y.

        Args:
            y (np.ndarray): known y values
            y_tilde (np.ndarray): y values as predicted by model

        Returns:
            float: calculated mean squared error
        """
        return np.sum((y - y_tilde) ** 2) / len(y)

    def _calculate_R2_score(self, y: np.ndarray, y_tilde: np.ndarray) -> float:
        """Calculates R2 score of predicted y_tilde comapred some known output 
        values y.

        Args:
            y (np.ndarray): known y values
            y_tilde (np.ndarray): y values as predicted by model

        Returns:
            float: calculated R2 score
        """
        # CHANGE THIS TO DEFINITION OF R2 SCORE!!!!! REMOVE WHEN DONE!!!!!!!
        return r2_score(y, y_tilde)

    def _compute_optimal_beta(self) -> None:
        """Raises error only if a model is incorrectly 
        instantiated through the superclass instead of 
        the subclasses. 

        Raises:
            NotImplementedError: method is not implemented 
            because something has been called incorrectly
        """
        raise NotImplementedError

    def _get_params(self, params: None | Iterable) -> Iterable:
        """Ensured parameters are iterable. Returns the given argument 
        if it is, and assigns self.params if not.

        Args:
            params (None | Iterable): None, or Iterable containing parameters 
            to be given to the model

        Returns:
            Iterable: Iterable containing parameters to be given to the model
        """
        if params is None:
            params = self.params
        elif not isinstance(params, Iterable):
            params = [params]
        return params

    def _plot_error_bias_variance(
        self, 
        error: np.ndarray, 
        bias: np.ndarray, 
        variance: np.ndarray, 
        functionof: np.ndarray, 
        titlestring: str, 
        y_limit: bool = False
    ) -> None:
        """Plots error, bias and variance in the same plot. Customizable x-axis.

        Args:
            error (np.ndarray): Error of model in an array for various cases
            bias (np.ndarray): Bias of model in an array for various cases
            variance (np.ndarray): Variance of model in an array for various cases
            functionof (np.ndarray): What the bias, variance and error are plotted as functions of
            titlestring (str): String to be added to the title of the plot describing what is being varied
            y_limit (bool, optional): Limits the top of the y-axis in the plot if True. Defaults to False.
        """
        plt.plot(functionof, error, label="Error")
        plt.plot(functionof, bias, label="Bias")
        plt.plot(functionof, variance, label="Variance")
        plt.title(f"Bias-variance tradeoff with varying {titlestring}")
        if y_limit:
            plt.ylim(top=1.5)
        plt.legend()
        plt.show()

# ------------------------------------------------------------- EXTERNAL METHODS ---------------------------------------------------------

    def calculate_MSE_across_degrees(self) -> None:
        """Fills dictionaries of MSE from training and 
        testing data with lists containing of MSE values. 
        Keys are the different parameter values, and each 
        parameter has MSE calculated for all degrees up
        to self.maxdegree.
        """
        for param in self.params:
            self.MSE_train[param] = np.empty(self.maxdegree)
            self.MSE_test[param] = np.empty(self.maxdegree)
            for degree in self.degrees:
                MSE_train = self._calculate_MSE(
                    self.y_train[degree], self.y_tilde_train[param][degree]
                )
                MSE_test = self._calculate_MSE(
                    self.y_test[degree], self.y_tilde_test[param][degree]
                )
                self.MSE_train[param][degree - 1] = MSE_train
                self.MSE_test[param][degree - 1] = MSE_test

    def calculate_R2_across_degrees(self) -> None:
        """Fills dictionaries of R2 scores from training 
        and testing data with lists containing of R2 scores. 
        Keys are the different parameter values, and each 
        parameter has R2 calculated for all degrees up
        to self.maxdegree.
        """
        for param in self.params:
            self.R2_score_train[param] = np.empty(self.maxdegree)
            self.R2_score_test[param] = np.empty(self.maxdegree)
            for degree in range(1, self.maxdegree + 1):
                MSE_train = self._calculate_R2_score(
                    self.y_train[degree], self.y_tilde_train[param][degree]
                )
                MSE_test = self._calculate_R2_score(
                    self.y_test[degree], self.y_tilde_test[param][degree]
                )
                self.R2_score_train[param][degree - 1] = MSE_train
                self.R2_score_test[param][degree - 1] = MSE_test

    def train_all_models(self) -> None:
        """Trains all models and fills dictionaries 
        of predicted values based on training and 
        testing data, as well as optimal coefficients 
        beta_hat.
        """
        for param in self.params:
            self.y_tilde_train[param] = {}
            self.y_tilde_test[param] = {}
            self.beta_hat[param] = {}

            for degree in self.degrees:
                X_train, y_train = self.X_train[degree], self.y_train[degree]
                beta_hat = self._compute_optimal_beta(X_train, y_train, param)
                y_tilde_train = self.X_train[degree] @ beta_hat
                y_tilde_test = self.X_test[degree] @ beta_hat
                if self.scale:
                    y_tilde_train += np.mean(self.y_train[degree])
                    y_tilde_test += np.mean(self.y_test[degree])
                self.y_tilde_train[param][degree] = y_tilde_train
                self.y_tilde_test[param][degree] = y_tilde_test
                self.beta_hat[param][degree] = beta_hat

    def plot_MSE_and_R2_scores(self, params: None | Iterable = None) -> None:
        """Plots MSE and R2 scores for the model as a function 
        of polynomial degree in the same plot. One plot is 
        generated per parameter since the comparison is to 
        polynomial degree.

        Args:
            params (None | Iterable, optional): Iterable containing 
            various parameters, each of which will generate a 
            plot. Will use self.params if nothing else is 
            specified. Defaults to None.
        """
        params = self._get_params(params)
        for param in params:
            fig, ax = plt.subplots()
            ax.plot(
                self.degrees,
                self.MSE_train[param],
                label="MSE of Training Data",
                linestyle="dotted",
                marker="o",
            )
            ax.plot(
                self.degrees,
                self.MSE_test[param],
                label="MSE of Testing Data",
                linestyle="dashed",
                marker="o",
            )
            ax.plot(
                self.degrees,
                self.R2_score_train[param],
                label=r"$R^2$ score of Training Data",
                linestyle="dotted",
                marker="o",
            )
            ax.plot(
                self.degrees,
                self.R2_score_test[param],
                label=r"$R^2$ Score of Testing Data",
                linestyle="dashed",
                marker="o",
            )
            ax.set_title(f"Statistical Metrics")  # of {self.name} Method")
            ax.set_xlabel("Degree of Polynomial Model")
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.set_ylabel("Value")
            ax.grid(True)
            ax.legend()
            if self.param_label is not None:
                fig.suptitle(f"{self.param_label} = {param}")
            plt.show()

    def plot_optimal_betas(self, params: None | Iterable = None) -> None:
        """Plots the optimal coefficients of the model that 
        are in self.beta_hat. If model has multiple parameters, 
        each of these will generate a separate plot.

        Args:
            params (None | Iterable, optional): Iterable containing 
            various parameters, each of which will generate a 
            plot. Will use self.params if nothing else is 
            specified. Defaults to None.
        """
        params = self._get_params(params)
        for param in params:
            fig, ax = plt.subplots()
            for degree, beta in self.beta_hat[param].items():
                ax.scatter([degree] * len(beta), beta)
                # ax.annotate(fr"\beta_{d+1}"+f" = {beta:.3f}", (d+1, beta), textcoords="offset points", xytext=(60, 7), ha='right')
            ax.set_title(f"Optimal Parameters of {self.name} Method")
            ax.set_xlabel("Degree of Polynomial Model")
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.set_ylabel("Value")
            ax.grid(True)
            if self.param_label is not None:
                fig.suptitle(f"{self.param_label} = {param}")
            plt.show()

    def bootstrap_resampling(self, num_bootstraps: int = 100, plot: bool = True) -> None | tuple[np.ndarray]:
        """Performs bias-variance analysis of dataset using bootstrap resampling.

        Args:
            num_bootstraps (int, optional): Number of bootstrap resamplings. Defaults to 100.
            plot (bool, optional): Boolean asserting whether to plot results or not. Defaults to True.

        Returns:
            None | tuple[np.ndarray]: If plot == True, nothing is returned and plots are shown. If plot == False, 
            a tuple is returned with arrays for error, bias and variance.
        """
        error = np.zeros(self.maxdegree)
        bias = np.zeros(self.maxdegree)
        variance = np.zeros(self.maxdegree)
        polydegree = np.zeros(self.maxdegree)

        X_train, X_test, y_train, y_test = self._split_train_test_data()

        for degree in self.degrees:

            y_pred = np.empty((y_test[degree].shape[0], num_bootstraps))

            for i in range(num_bootstraps):
                X_, y_ = resample(X_train[degree], y_train[degree])

                # Evaluating the new model on the same test data every time
                beta_hat = self._compute_optimal_beta(X_, y_)
                y_tilde = X_test[degree] @ beta_hat + np.mean(y_)
                y_pred[:, i] = y_tilde.ravel()

            error[degree - 1] = np.mean(
                np.mean((y_test[degree] - y_pred) ** 2, axis=1, keepdims=True)
            )
            bias[degree - 1] = np.mean(
                (y_test[degree] - np.mean(y_pred, axis=1, keepdims=True)) ** 2
            )
            variance[degree - 1] = np.mean(np.var(y_pred, axis=1, keepdims=True))

        if plot:
            self._plot_error_bias_variance(
                error, bias, variance, self.degrees, "complexity of polynomial model"
            )
        else:
            return error, bias, variance

    def plot_bootstrap_per_degree(self, num_bootstraps):
        # More code here...

        # error, bias, variance = self.bootstrap_resampling(...)
        # self._plot_error_bias_variance(error, bias, variance, self.degrees, "complexity of polynomial model")
        raise NotImplementedError

    def plot_bootstrap_per_datasize(self, num_bootstraps):
        # More code here...

        # error, bias, variance = self.bootstrap_resampling(...)
        # self._plot_error_bias_variance(error, bias, variance, self.degrees, "size of data set")
        raise NotImplementedError

    def plot_bootstrap_per_numbootstrap(self, num_bootstraps):
        # More code here...

        # error, bias, variance = self.bootstrap_resampling(...)
        # self._plot_error_bias_variance(error, bias, variance, self.degrees, "number of bootstrap resamplings")
        raise NotImplementedError

    def kfold_cross_validation(self, k:  int, degree: int, param: float) -> float:
        """Performs K-Fold cross validation resampling.

        Args:
            k (int): number of folds
            degree (int): degree of model to look at
            param (float): parameter to be given to model

        Returns:
            float: mean squared error after cross validation
        """
        X, Y = self.design_matrices[degree], self.Y
        kfold = KFold(n_splits=k)
        scores_kfold = np.empty(k)
        i = 0
        for train_inds, test_inds in kfold.split(X):
            X_train = X[train_inds]
            y_train = Y[train_inds]

            X_test = X[test_inds]
            y_test = Y[test_inds]

            if self.scale:
                X_train = scale(X_train, with_std=False)
                X_test = scale(X_test, with_std=False)

            beta_hat = self._compute_optimal_beta(X_train, y_train, param)
            y_pred_test = X_test @ beta_hat

            if self.scale:
                y_pred_test += np.mean(y_test)

            MSE_test = self._calculate_MSE(y_test, y_pred_test)
            scores_kfold[i] = MSE_test
            i += 1
        return np.mean(scores_kfold)

    def plot_kfold_per_degree(self, k: int, params: None | Iterable = None) -> None:
        """Plots mean squared error as a function of polynomial degree after a cross 
        validation resampling has been performed with k folds.

        Args:
            k (int): number of folds to use in K-Fold cross validation
            params (None | Iterable, optional): Iterable containing 
            various parameters, each of which will generate a 
            plot. Will use self.params if nothing else is 
            specified. Defaults to None. Defaults to None.
        """
        params = self._get_params(params)
        for param in params:
            fig, ax = plt.subplots()
            estimated_MSE_kfold = np.empty(self.maxdegree)
            for degree in self.degrees:
                estimated_MSE_kfold[degree - 1] = self.kfold_cross_validation(
                    k, degree, param
                )
            ax.plot(self.degrees, estimated_MSE_kfold)
            ax.set_title(f"K-fold Cross Validation of {self.name} Method")
            ax.set_xlabel("Degree of Polynomial Model")
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.set_ylabel("MSE")
            if self.param_label is not None:
                fig.suptitle(f"{self.param_label} = {param}")
            plt.show()

    def plot_kfold_per_param(self, k: int, degree: None | int = None, params: None | Iterable = None) -> None:
        """_summary_

        Args:
            k (int): number of folds to use in K-Fold cross validation
            degree (None | int, optional): Degree to plot MSE for. Uses self.maxdegree if none is given. Defaults to None.
            params (None | Iterable, optional): Iterable containing various parameters. Will use self.params if nothing else 
            is specified. Defaults to None. 
        """
        if degree is None:
            degree = self.maxdegree
        if self.name == "Ordinary Least Squares":
            return  # raise an error?
        fig, ax = plt.subplots()
        params = self._get_params(params)
        estimated_MSE_kfold = np.empty(len(params))
        for i, param in enumerate(params):
            estimated_MSE_kfold[i] = self.kfold_cross_validation(k, degree, param)
        ax.plot(np.log10(params), estimated_MSE_kfold, label="KFold")
        ax.legend()
        ax.set_title(f"K-fold Cross Validation of {self.name} Method")
        ax.set_xlabel(self.param_label)
        ax.set_ylabel("MSE")
        plt.show()


class OrdinaryLeastSquares(Model):

    def __init__(
        self,
        X: np.ndarray | tuple[np.ndarray],
        Y: np.ndarray,
        maxdegree: int,
        test_size: float = 0.2,
        scale: bool = True,
        multidim: bool = False,
    ) -> None:
        """Initializes an instance of an Ordinary Least Squares model.

        Args:
            X (np.ndarray | tuple[np.ndarray]): Input data
            Y (np.ndarray): Output data
            maxdegree (int): Maximum degree of polynomial model to create. The class will create models up to and including this degree.
            test_size (float, optional): Ratio of data to be set aside for testing. Defaults to 0.2.
            scale (bool, optional): Whether or not to scale the data. Defaults to True.
            multidim (bool, optional): Whether or not the dataset has multivariate inputs. Defaults to False.
        """

        param = [None]
        super().__init__(X, Y, maxdegree, param, test_size, scale, multidim)
        self.colors = [
            "royalblue",
            "cornflowerblue",
            "chocolate",
            "sandybrown",
            "orchid",
        ]
        self.name = "Ordinary Least Squares"

    def _compute_optimal_beta(self, X_train: np.ndarray, y_train: np.ndarray, param: float = None) -> np.ndarray:
        """Finds optimal coefficients for polynomial model for one degree.

        Args:
            X_train (np.ndarray): Design matrix of training data.
            y_train (np.ndarray): Output values from training data
            param (float, optional): Parameter to give to cost function, but does nothing for OLS. Defaults to None.

        Returns:
            np.ndarray: Optimal coefficients of model in an array.
        """
        beta_hat = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
        return beta_hat


class RidgeRegression(Model):

    def __init__(
        self,
        X: np.ndarray | tuple[np.ndarray],
        Y: np.ndarray,
        maxdegree: int,
        lmbdas: list | int,
        test_size: float = 0.2,
        scale: bool = True,
        multidim: bool = False,
    ) -> None:
        """Initializes an instance of a Ridge Regression model.

        Args:
            X (np.ndarray | tuple[np.ndarray]): Input data
            Y (np.ndarray): Output data
            maxdegree (int): Maximum degree of polynomial model to create. The class will create models up to and including this degree.
            lmbdas (list | int): list or integer with parameter to be given to cost function of model.
            test_size (float, optional): Ratio of data to be set aside for testing. Defaults to 0.2.
            scale (bool, optional): Whether or not to scale the data. Defaults to True.
            multidim (bool, optional): Whether or not the dataset has multivariate inputs. Defaults to False.
        """

        if isinstance(lmbdas, int):
            lmbdas = [lmbdas]
        super().__init__(X, Y, maxdegree, lmbdas, test_size, scale, multidim)
        self.colors = [
            "forestgreen",
            "limegreen",
            "darkgoldenrod",
            "goldenrod",
            "darkorange",
        ]
        self.name = "Ridge Regression"
        self.param_label = r"$\lambda$"

    def _compute_optimal_beta(self, X_train: np.ndarray, y_train: np.ndarray, lmbda: float) -> np.ndarray:
        """Finds optimal coefficients for polynomial model for one degree.

        Args:
            X_train (np.ndarray): Design matrix of training data.
            y_train (np.ndarray): Output values from training data.
            lmbda (float): Parameter to give to cost function.

        Returns:
            np.ndarray: Optimal coefficients of model in an array.
        """
        p = len(X_train[0])
        beta_hat = (
            np.linalg.inv(X_train.T @ X_train + lmbda * np.eye(p)) @ X_train.T @ y_train
        )
        return beta_hat


class LassoRegression(Model):

    def __init__(
        self,
        X: np.ndarray | tuple[np.ndarray],
        Y: np.ndarray,
        maxdegree: int,
        alphas: list | int,
        test_size: float = 0.2,
        scale: bool = True,
        multidim: bool = False,
    ) -> None:
        """Initializes an instance of a Lasso Regression model.

        Args:
            X (np.ndarray | tuple[np.ndarray]): Input data
            Y (np.ndarray): Output data
            maxdegree (int): Maximum degree of polynomial model to create. The class will create models up to and including this degree.
            alphas (list | int): list or integer with parameter to be given to cost function of model.
            test_size (float, optional): Ratio of data to be set aside for testing. Defaults to 0.2.
            scale (bool, optional): Whether or not to scale the data. Defaults to True.
            multidim (bool, optional): Whether or not the dataset has multivariate inputs. Defaults to False.
        """

        if isinstance(alphas, int):
            alphas = [alphas]
        super().__init__(X, Y, maxdegree, alphas, test_size, scale, multidim)
        self.colors = [
            "firebrick",
            "lightcoral",
            "lightseagreen",
            "turquoise",
            "blueviolet",
        ]
        self.name = "Lasso Regression"
        self.param_label = r"$\alpha$"

    def _compute_optimal_beta(self, X_train: np.ndarray, y_train: np.ndarray, alpha: float) -> np.ndarray:
        """Finds optimal coefficients for polynomial model for one degree.

        Args:
            X_train (np.ndarray): Design matrix of training data.
            y_train (np.ndarray): Output values from training data.
            alpha (float): Parameter to give to cost function.

        Returns:
            np.ndarray: Optimal coefficients of model in an array.
        """
        beta_hat = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

        for b in enumerate(beta_hat):
            if b[1] > alpha / 2:
                beta_hat[b[0]] = b[1] - alpha / 2
            elif b[1] < -alpha / 2:
                beta_hat[b[0]] = b[1] + alpha / 2
            elif abs(b[1]) <= alpha / 2:
                beta_hat[b[0]] = 0

        return beta_hat


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
        Franke += np.random.normal(0, 0.1, x.shape)

    return Franke

def generate_data(n: int, seed: int, multidim: bool = False) -> tuple[np.ndarray]:
    """Generates noisy data to be given to our model. Can give multivariate or univariate data.

    Args:
        n (int): number of data points
        seed (int): set seed for consistency with random noise
        multidim (bool, optional): Whether or not to make the data multivariate. Defaults to False.

    Returns:
        tuple[np.ndarray]: Input and output data in a tuple. If multivariate, input data is itself 
        a tuple of various inputs X1 and X2.
    """
    np.random.seed(seed)
    if multidim:
        x1 = np.linspace(0, 1, n).reshape(-1, 1)
        x2 = np.linspace(0, 1, n).reshape(-1, 1)
        X1, X2 = np.meshgrid(x1, x2)
        Y = Franke_function(X1, X2)
        return (X1, X2), Y

    else:
        x = np.linspace(-3, 3, n).reshape(-1, 1)
        y = (
            np.exp(-(x**2))
            + 1.5 * np.exp(-((x - 2) ** 2))
            + np.random.normal(0, 0.1, x.shape)
        )
        return x, y

def process_irl_data(multidim=False):
    if multidim:
        # 2D data set
        file_path_2d = "datasets/ClassicDisco.csv"
        df  = pd.read_csv(file_path_2d)

        input1 = "Duration"
        input2 = "Energy"
        output = "Danceability"

        filtered_df = df[[input1, input2, output]].apply(pd.to_numeric, errors='coerce').dropna()

        X1 = filtered_df[input1].to_numpy().reshape(-1,1)
        X2 = filtered_df[input1].to_numpy().reshape(-1,1)
        Y = filtered_df[output].to_numpy().reshape(-1,1) 

        X = (X1, X2)

    else:
       # 1D data set
        file_path_1d = "datasets/archive/epa-sea-level.csv"
        df = pd.read_csv(file_path_1d)

        input_column = "Year"
        output_column = "CSIRO Adjusted Sea Level"  

        filtered_df = df[[input_column, output_column]].apply(pd.to_numeric, errors='coerce').dropna()

        X = filtered_df[input_column].to_numpy().reshape(-1,1)
        Y = filtered_df[output_column].to_numpy().reshape(-1,1) 
    
    return X, Y

def main():
    n = 100
    seed = 8
    test_size = 0.2
    maxdegree = 6
    scale = True
    multidim = True

    params = [0.0001, 0.001, 0.01, 0.1, 1.0]

    #X, Y = generate_data(n, seed, multidim)

    X, Y = process_irl_data(multidim=multidim)

    OLS = OrdinaryLeastSquares(X, Y, maxdegree, test_size, scale, multidim)
    OLS.train_all_models()
    OLS.calculate_MSE_across_degrees()
    OLS.calculate_R2_across_degrees()
    OLS.plot_MSE_and_R2_scores()
    OLS.plot_optimal_betas()
    OLS.plot_kfold_per_degree(5)

    # Bias-variance analysis of OLS with bootstrap:
    OLS.bootstrap_resampling()

    Ridge = RidgeRegression(
        X, Y, maxdegree, params, scale=scale, multidim=multidim, test_size=test_size
    )
    Ridge.train_all_models()
    Ridge.calculate_MSE_across_degrees()
    Ridge.calculate_R2_across_degrees()
    Ridge.plot_MSE_and_R2_scores(0.1)
    Ridge.plot_optimal_betas()
    Ridge.plot_kfold_per_degree(5)
    Ridge.plot_kfold_per_param(5)
    # Ridge.MSE_per_lmbda(10)

    Lasso = LassoRegression(
        X, Y, maxdegree, params, scale=scale, multidim=multidim, test_size=test_size
    )
    Lasso.train_all_models()
    Lasso.calculate_MSE_across_degrees()
    Lasso.calculate_R2_across_degrees()
    Lasso.plot_MSE_and_R2_scores()
    Lasso.plot_optimal_betas()
    Lasso.plot_kfold_per_degree(5)
    Lasso.plot_kfold_per_param(5)
    # Lasso.MSE_per_alpha(10)


if __name__ == "__main__":
    main()
