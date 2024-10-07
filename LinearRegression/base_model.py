import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class BaseModel:
    def __init__(
            self, 
            degree: int, 
            multidim: bool = True,
            ) -> None:
        """
        Initialize the BaseModel.

        Args:
            degree (int): The degree of the polynomial for fitting.
            multidim (bool, optional): Whether to use a 2D polynomial. Default is True.
        """
        self.degree = degree
        self.multidim = multidim

        # Number of features
        if multidim:
            self._num_col = int((degree + 1) * (degree + 2) / 2 - 1)
        else:
            self._num_col = degree

        self.scale = False
        self.scaler = None
        self._y_train_mean = None
        self.beta_hat = None

    def create_design_matrix(self, x: np.ndarray | tuple[np.ndarray]) -> np.ndarray:
        """
        Generate a design matrix from the input data for a polynomial fit of the degree 
        specified in the constructor.

        Args:
            x (np.ndarray or tuple[np.ndarray, np.ndarray]]): 
                - If `self.multidim` is True: A tuple containing two 2D arrays (x1, x2), 
                each of shape (n, 1), where n is the number of data points.
                - If `self.multidim` is False: A single 2D array with shape (n, 1).

        Returns:
            np.ndarray: The design matrix corresponding to the polynomial degree specified in the constructor. 
            The output matrix will be univariate if `self.multidim` is False, 
            or multivariate if `self.multidim` is True.
        """
        if self.multidim:
            x1, x2 = x

            n = len(x1)

            X = np.zeros((n, self._num_col))

            for i in range(self.degree):
                # Calculate base index for the current degree
                base_index = int(
                    (i + 1)*(i + 2)/2 - 1
                )  

                # Fill the design matrix with x and y raised to appropriate powers
                for j in range(i + 2):
                    X[:, base_index + j] = x1[:,0]**(i + 1 - j) * x2[:,0]**(j)
            return X

        else:
            X = np.zeros((len(x), self.degree))

            for i in range(self.degree):
                X[:, i] = x[:, 0]**(i + 1)
            return X

    def split_test_train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> tuple:
        """
        Split arrays or matrices into random train and test subsets, with size of test set 
        specified in the constructor.

        Args:
            X (np.ndarray): Design matrix.
            y (np.ndarray): Output data.

        Returns:
            tuple[np.ndarray]: X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return X_train, X_test, y_train, y_test

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, with_std: bool = True) -> np.ndarray:
        """
        Feature scale the `X_train` by subtracting the mean and dividing by the standard deviation (std)
        for each meature. 
        Store the feature mean and std in class for use when scaling other matrices.

        Args:
            X_train (np.ndarray): The training matrix for this model.
            y_train (np.ndarray): The training output.

        Returns:
            (np.ndarray): The scaled training matrix, where each feature has been standardized to 
            have a mean of 0 and a standard deviation of 1.
        """
        self.scale = True
        self.scaler = StandardScaler(with_std=with_std)
        self.scaler.fit(X_train)
        self._y_train_mean = np.mean(y_train)

    def transform(self, X):
        """
        Scale `X` with same mean (and standard deviation) as `X_train`.

        Args:
            X (np.ndarray): Matrix to be scaled.

        Returns:
            (np.ndarray): Scaled matrix
        """
        return self.scaler.transform(X)

    def train(self, X_train, y_train):
        """Raises error only if a model is incorrectly 
        instantiated through the superclass instead of 
        the subclasses. 

        Raises:
            NotImplementedError: method is not implemented 
            because something has been called incorrectly
        """
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input data using the model's learned parameters.
        Adds mean of `y_train` if scaling is used. 

        Args:
            X (np.ndarray): Input data for prediction.

        Returns:
            np.ndarray: Predicted output values.
        """
        y_pred = X @ self.beta_hat
        if self.scale:
            y_pred += self._y_train_mean
        return y_pred

