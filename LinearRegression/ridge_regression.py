import numpy as np

from .base_model import BaseModel


class RidgeRegression(BaseModel):
    def __init__(self, degree, param, multidim=True):
        """Initializes an instance of Ridge Regression model.

        Args:
            degree (int): Degree of polynomial model to create.
            lmbda (int | float): Hyper parameter.
            multidim (bool, optional): Whether or not the dataset has multivariate. Defaults to True.
            scale (bool, optional): Whether or not to scale the data. Defaults to True.
        """
        super().__init__(degree, multidim)
        self.lmbda = param
        self.name = "Ridge Regression"
        self.param_label = r"$\lambda$"

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Finds optimal coefficients for polynomial model for one degree.

        Args:
            X_train (np.ndarray): Design matrix of training data.
            y_train (np.ndarray): Output values from training data.

        Returns:
            np.ndarray: Optimal coefficients of model in an array.
        """
        beta_hat = (
            np.linalg.inv(X_train.T @ X_train + self.lmbda * np.eye(self._num_col)) @ X_train.T @ y_train
        )
        self.beta_hat = beta_hat
        return beta_hat