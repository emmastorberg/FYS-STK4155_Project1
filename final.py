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



class Model:

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
        # May have X, Y, Z when all is multivairate
        self.X = X
        self.Y = Y.flatten()
        self.maxdegree = maxdegree
        self.degrees = np.arange(1, maxdegree + 1)
        self.params = params
        self.test_size = test_size
        self.scale = scale
        self.multidim = multidim
        self.design_matrices = self._generate_design_matrices()
        self.X_train, self.X_test, self.y_train, self.y_test = self._split_train_test_data()

        self.beta_hat = {}
        self.y_tilde_train = {}
        self.y_tilde_test = {}
        self.MSE_train = {}
        self.MSE_test = {}
        self.R2_score_train = {}
        self.R2_score_test = {}
        self.param_label = None

    def _generate_design_matrices(self):
        design_matrices = {}
        for degree in self.degrees:
            design_matrices[degree] = self._create_design_matrix(degree=degree)
        return design_matrices

    def _split_train_test_data(self):
        X_train = {}
        X_test = {}
        y_train = {}
        y_test = {}
        for d, X in self.design_matrices.items():
            X_train[d], X_test[d], y_train[d], y_test[d] = train_test_split(X, self.Y, test_size=self.test_size)
            if self.scale:
                X_train[d] = scale(X_train[d], with_std=False)
                X_test[d] = scale(X_test[d], with_std=False)
        return X_train, X_test, y_train, y_test

    def _create_design_matrix(self, degree) -> np.ndarray:
        if self.multidim:
            # multivariate
            X1, X2 = self.X
            X1 = X1.flatten()
            X2 = X2.flatten()

            n = len(X1)
            p = int((degree + 1) * (degree + 2) / 2 - 1)  # Number of terms in the polynomial expansion

            X = np.zeros((n, p))

            for i in range(degree):
                base_index = int((i + 1) * (i + 2) / 2 - 1)  # Calculate base index for the current degree

                for j in range(i + 2):
                    # Fill the design matrix with x and y raised to appropriate powers
                    X[:, base_index + j] = X1**(i + 1 - j) * X2**(j)
            return X

        else:
            # univariate
            X = np.zeros((len(self.X), degree))

            for i in range(degree):
                X[:,i] = self.X[:,0]**(i + 1)
            return X

    def _calculate_MSE(self, y, y_tilde):
        return np.sum((y - y_tilde)**2) / len(y)
    
    def _calculate_R2_score(self, y, y_tilde):
        return r2_score(y, y_tilde)

    def calculate_MSE_across_degrees(self):
        for param in self.params:
            self.MSE_train[param] = np.empty(self.maxdegree)
            self.MSE_test[param] = np.empty(self.maxdegree)
            for degree in self.degrees:
                MSE_train = self._calculate_MSE(self.y_train[degree], self.y_tilde_train[param][degree])
                MSE_test = self._calculate_MSE(self.y_test[degree], self.y_tilde_test[param][degree])
                self.MSE_train[param][degree-1] = MSE_train
                self.MSE_test[param][degree-1] = MSE_test

    def calculate_R2_across_degrees(self):
        for param in self.params:
            self.R2_score_train[param] = np.empty(self.maxdegree)
            self.R2_score_test[param] = np.empty(self.maxdegree)
            for degree in range(1, self.maxdegree + 1):
                MSE_train = self._calculate_R2_score(self.y_train[degree], self.y_tilde_train[param][degree])
                MSE_test = self._calculate_R2_score(self.y_test[degree], self.y_tilde_test[param][degree])
                self.R2_score_train[param][degree-1] = MSE_train
                self.R2_score_test[param][degree-1] = MSE_test
    
    def _compute_optimal_beta(self):
        raise NotImplementedError

    def train_all_models(self):
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

    def plot_MSE_and_R2_scores(self, params=None):
        params = self._get_params(params)
        for param in params:
            fig, ax = plt.subplots()
            ax.plot(self.degrees, self.MSE_train[param], label="MSE of Training Data", linestyle="dotted", marker="o")
            ax.plot(self.degrees, self.MSE_test[param], label="MSE of Testing Data", linestyle="dashed", marker="o")
            ax.plot(self.degrees, self.R2_score_train[param], label=r"$R^2$ score of Training Data", linestyle="dotted", marker="o")
            ax.plot(self.degrees, self.R2_score_test[param], label=r"$R^2$ Score of Testing Data", linestyle="dashed", marker="o")
            ax.set_title(f"Statistical Metrics")# of {self.name} Method")
            ax.set_xlabel("Degree of Polynomial Model")
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.set_ylabel("Value")
            ax.grid(True)
            ax.legend()
            if self.param_label is not None:
                fig.suptitle(f"{self.param_label} = {param}")
            plt.show()

    def _get_params(self, params):
        """Returns self.params if params is None, and make sure params is iterable."""
        if params is None:
            params = self.params
        elif not isinstance(params, Iterable):
            params = [params]
        return params
    
    def plot_optimal_betas(self, params=None):
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

    def kfold_cross_validation(self, k, degree, param):
        X,Y = self.design_matrices[degree], self.Y
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
    
    def plot_kfold_per_degree(self, k, params=None):
        params = self._get_params(params)
        for param in params:
            fig, ax = plt.subplots()
            estimated_MSE_kfold = np.empty(self.maxdegree)
            for degree in self.degrees:
                estimated_MSE_kfold[degree-1] = self.kfold_cross_validation(k, degree, param)
            ax.plot(self.degrees, estimated_MSE_kfold)
            ax.set_title(f"K-fold Cross Validation of {self.name} Method")
            ax.set_xlabel("Degree of Polynomial Model")
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.set_ylabel("MSE")
            if self.param_label is not None:
                fig.suptitle(f"{self.param_label} = {param}")
            plt.show()

    def plot_kfold_per_param(self, k, degree=None, params=None):
        if degree is None:
            degree = self.maxdegree
        if self.name == "Ordinary Least Squares":
            return # raise an error?
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

        param = [None]
        super().__init__(X, Y, maxdegree, param, test_size, scale, multidim)
        self.colors = ["royalblue", "cornflowerblue", "chocolate", "sandybrown","orchid"]
        self.name = "Ordinary Least Squares"

    def _compute_optimal_beta(self, X_train, y_train, param=None):
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

        if isinstance(lmbdas, int):
            lmbdas = [lmbdas]
        super().__init__(X, Y, maxdegree, lmbdas, test_size, scale, multidim)
        self.colors = ["forestgreen", "limegreen", "darkgoldenrod", "goldenrod", "darkorange"]
        self.name = "Ridge Regression"
        self.param_label = r"$\lambda$"

    def _compute_optimal_beta(self, X_train, y_train, lmbda):
        p = len(X_train[0])
        beta_hat = np.linalg.inv(X_train.T @ X_train + lmbda * np.eye(p)) @ X_train.T @ y_train
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

        if isinstance(alphas, int):
            alphas = [alphas]
        super().__init__(X, Y, maxdegree, alphas, test_size, scale, multidim)
        self.colors = ["firebrick", "lightcoral", "lightseagreen", "turquoise", "blueviolet"]
        self.name = "Lasso Regression"
        self.param_label = r"$\alpha$"

    def _compute_optimal_beta(self, X_train, y_train, alpha):
        lasso = Lasso(alpha=alpha, fit_intercept=False)
        lasso.fit(X_train, y_train)
        beta_hat = lasso.coef_
        return beta_hat
    

def Franke_function(x: np.ndarray, y: np.ndarray, noise: bool = True) -> np.ndarray:
    """
    Calculate the Franke function for 
    """
    term1 = 3/4 * np.exp(-((9*x - 2)**2)/4 - ((9*y - 2)**2)/4)
    term2 = 3/4 * np.exp(-((9*x + 1)**2)/49 - (9*y + 1)/10)
    term3 = 1/2 * np.exp(-((9*x - 7)**2)/4 - ((9*y - 3)**2)/4)
    term4 = - 1/5 * np.exp(-(9*x - 4)**2 - (9*y - 7)**2)

    Franke = term1 + term2 + term3 + term4

    if noise:
        Franke += np.random.normal(0, 0.1, x.shape)

    return Franke


def generate_data(n: int, seed: int, multidim: bool = False) -> tuple[np.ndarray]:
    np.random.seed(seed)
    if multidim:
        x1 = np.linspace(0, 1, n).reshape(-1, 1)
        x2 = np.linspace(0, 1, n).reshape(-1, 1)
        X1, X2 = np.meshgrid(x1, x2)
        Y = Franke_function(X1, X2)
        return (X1, X2), Y

    else:
        x = np.linspace(-3, 3, n).reshape(-1, 1)
        y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)
        return x, y


def main():
    n = 100
    seed = 8
    test_size = 0.2
    maxdegree = 6
    scale = True
    multidim = True

    lmbda = [0.0001, 0.001, 0.01, 0.1, 1.0]
    alpha = [0.1, 0.5, 0.9, 1.5, 3.0]

    X, Y = generate_data(n, seed, multidim)

    OLS = OrdinaryLeastSquares(X, Y, maxdegree, test_size, scale, multidim)
    OLS.train_all_models()
    OLS.calculate_MSE_across_degrees()
    OLS.calculate_R2_across_degrees()
    OLS.plot_MSE_and_R2_scores()
    OLS.plot_optimal_betas()
    OLS.plot_kfold_per_degree(5)

    Ridge = RidgeRegression(X, Y, maxdegree, lmbda, scale=scale, multidim=multidim, test_size=test_size)
    Ridge.train_all_models()
    Ridge.calculate_MSE_across_degrees()
    Ridge.calculate_R2_across_degrees()
    Ridge.plot_MSE_and_R2_scores(0.1)
    Ridge.plot_optimal_betas([0.001, 1.0])
    Ridge.plot_kfold_per_degree(5)
    Ridge.plot_kfold_per_param(5)
    # Ridge.MSE_per_lmbda(10)

    Lasso = LassoRegression(X, Y, maxdegree, alpha, scale=scale, multidim=multidim, test_size=test_size)
    Lasso.train_all_models()
    Lasso.calculate_MSE_across_degrees()
    Lasso.calculate_R2_across_degrees()
    Lasso.plot_MSE_and_R2_scores([3.0, 0.1])
    Lasso.plot_optimal_betas(0.5)
    Lasso.plot_kfold_per_degree(5)
    Lasso.plot_kfold_per_param(5)
    # Lasso.MSE_per_alpha(10)
    

if __name__ == "__main__":
    main()
