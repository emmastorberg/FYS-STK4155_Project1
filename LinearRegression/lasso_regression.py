import numpy as np
from sklearn.linear_model import Lasso

from .base_model import BaseModel


class LassoRegression(BaseModel):
    def __init__(self, degree, param, multidim=True):
        """Initializes an instance of Lasso Regression model.

        Args:
            degree (int): Degree of polynomial model to create.
            lmbda (int | float): Hyper parameter.
            multidim (bool, optional): Whether or not the dataset has multivariate. Defaults to True.
            scale (bool, optional): Whether or not to scale the data. Defaults to True.
        """
        super().__init__(degree, multidim)
        self.alpha = param
        self.name = "Lasso Regression"
        self.param_label = r"$\alpha$"

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Finds optimal coefficients for polynomial model for one degree.

        Args:
            X_train (np.ndarray): Design matrix of training data.
            y_train (np.ndarray): Output values from training data.

        Returns:
            np.ndarray: Optimal coefficients of model in an array.
        """
        linreg = Lasso(alpha=self.alpha, fit_intercept=False)
        linreg.fit(X_train, y_train)
        beta_hat = linreg.coef_.reshape(-1, 1)
        self.beta_hat = beta_hat
        return beta_hat