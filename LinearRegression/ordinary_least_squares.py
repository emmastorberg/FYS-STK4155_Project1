import numpy as np

from .base_model import BaseModel


class OrdinaryLeastSquares(BaseModel):
    def __init__(self, degree, param=None, multidim=True):
        """Initializes an instance of an Ordinary Least Squares model.

        Args:
            degree (int): Degree of polynomial model to create.
            multidim (bool, optional): Whether or not the dataset has multivariate. Defaults to True.
            scale (bool, optional): Whether or not to scale the data. Defaults to True.
        """
        super().__init__(degree, multidim)
        self.name = "Ordinary Least Squares"
        self.param_label = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Finds optimal coefficients for polynomial model for one degree.

        Args:
            X_train (np.ndarray): Design matrix of training data.
            y_train (np.ndarray): Output values from training data
            param (float, optional): Parameter to give to cost function, but does nothing for OLS. Defaults to None.

        Returns:
            np.ndarray: Optimal coefficients of model in an array.
        """
        beta_hat = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
        self.beta_hat = beta_hat
        return beta_hat