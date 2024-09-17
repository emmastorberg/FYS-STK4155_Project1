import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import cm
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class Model:
    def __init__(
        self, 
        x: np.ndarray | tuple[np.ndarray],
        y: np.ndarray,
        maxdegree: int,
        test_size: float,
        scale: bool,
        multidim: bool,
        ) -> None:

        self.x = x
        self.y = y
        self.maxdegree = maxdegree
        self.test_size = test_size
        self.scale = scale
        self.multidim = multidim
        self.design_matrices = self.design_matrix_dict()
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_dicts()

    def design_matrix_dict(self):
        design_matrices = {}
        for d in range(self.maxdegree):
            design_matrices[d] = self.design_matrix(degree=d+1)
        return design_matrices

    def train_test_dicts(self):
        X_train = {}
        X_test = {}
        y_train = {}
        y_test = {}
        for d, X in self.design_matrices.items():
            X_train[d], X_test[d], y_train[d], y_test[d] = train_test_split(X, self.y, test_size=self.test_size)
        return X_train, X_test, y_train, y_test


    def design_matrix(self, degree) -> np.ndarray:
        """
        Generates a design matrix for polynomial fitting (univariate).

        Args:
            x (np.ndarray): Array of x values, shape (n,1).
            degree (int): Degree of the polynomial.
            scale (bool): Whether to scale the data or not.

        Returns:
            (np.ndarray): The generated design matrix for the polynomial fit. shape (n, p).
        """
        if self.multidim:
            # multivariate
            x1, x2 = self.x
            n = len(x1)
            p = int((degree + 1) * degree / 2 + degree)  # Number of terms in the polynomial expansion

            X = np.zeros((n, p))
            C = np.zeros(p)

            for i in range(degree):
                base_index = int((i + 1) * i / 2 + i)  # Calculate base index for the current degree

                for j in range(i + 2):
                    # Fill the design matrix with x and y raised to appropriate powers
                    X[:, base_index + j] = x1[:, 0]**(i + 2 - j) * x2[:, 0]**(j)
                    if self.scale:
                        X[:, base_index + 1] -= np.mean(X[:, base_index + 1])
                    C[base_index + j] = math.comb(i + 1, j)  # Calculate binomial coefficient

            return X

        else:
            # univariate
            X = np.zeros((len(self.x), degree))

            # WE MUST COMMENT ON THE SCALING! REMOVE THIS WHEN DONE
            for i in range(degree):
                X[:,i] = self.x[:,0]**(i + 1)
                if self.scale:
                    X[:,i] -= np.mean(X[:,i])
            return X

    def plot_train_test_and_parameters(self, param = None):
            colors = ["royalblue", "cornflowerblue", "chocolate", "sandybrown","orchid"]
                
            # Plotting statistical metrics
            fig, axs = plt.subplots(1, 2, figsize=(16, 9))
            degrees = np.arange(1, self.maxdegree + 1)

            if param is None:
                MSE_train = self.MSE_train
                MSE_test = self.MSE_test
                R2_train = self.R2_train
                R2_test = self.R2_test
                beta_hat = self.beta_hat
            else:
                MSE_train = self.MSE_train[param]
                MSE_test = self.MSE_test[param]
                R2_train = self.R2_train[param]
                R2_test = self.R2_test[param]
                beta_hat = self.beta_hat[param]

            axs[0].plot(degrees, MSE_train, label="MSE of Training Data", color=colors[0], linestyle="dotted", marker="o")
            axs[0].plot(degrees, MSE_test, label="MSE of Testing Data", color=colors[1], linestyle="dashed", marker="o")
            axs[0].plot(degrees, R2_train, label=r"$R^2$ Score of Training Data", color=colors[2], linestyle="dotted", marker="o")
            axs[0].plot(degrees, R2_test, label=r"$R^2$ Score of Testing Data", color=colors[3], linestyle="dashed", marker="o")
            axs[0].set_title("Statistical Metrics")
            axs[0].set_xlabel("Degree of Polynomial Model")
            axs[0].xaxis.set_major_locator(MultipleLocator(1))
            axs[0].set_ylabel("Value")
            axs[0].legend()
            axs[0].grid(True)

            # Plotting optimal parameters
            axs[1].scatter(degrees, beta_hat, color=colors[4])

            """
            This labeling looks a bit wonky. Fix later if necessary.
            """
            for degree, b in zip(degrees, beta_hat):
                axs[1].annotate(fr"\beta_{degree}"+f" = {b:.3f}", (degree, b), textcoords="offset points", xytext=(60, 7), ha='right')

            axs[1].set_title("Optimal Parameters")
            axs[1].set_xlabel("Degree of Polynomial Model")
            axs[1].xaxis.set_major_locator(MultipleLocator(1))
            axs[1].set_ylabel("Value")
            axs[1].grid(True)

            fig.suptitle(self.name)
            plt.tight_layout()
            plt.show()


class OrdinaryLeastSquares(Model):
    def __init__(
        self, 
        x: np.ndarray | tuple[np.ndarray],
        y: np.ndarray,
        maxdegree: int,
        test_size: float = 0.2,
        scale: bool = True,
        multidim: bool = False,
        ) -> None:

        super().__init__(x, y, maxdegree, test_size, scale, multidim)
        self.colors = ["royalblue", "cornflowerblue", "chocolate", "sandybrown","orchid"]
        self.name = "Ordinary Least Squares"

    def predict(self):
        maxdegree = self.maxdegree

        self.y_tilde_train = {}
        self.y_tilde_test = {}
        self.beta_hat = {}
        
        for d in range(maxdegree):
            X_train = self.X_train[d]
            X_test = self.X_test[d]
            y_train = self.y_train[d]
            print(X_train.T @ X_train)
            self.beta_hat[d] = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
            beta_hat = self.beta_hat[d] 

            # WE MUST COMMENT ON THE SCALING! REMOVE THIS WHEN DONE
            if self.scale:
                self.y_tilde_train[d] = X_train @ beta_hat+ np.mean(y_train)
                self.y_tilde_test[d] = X_test @ beta_hat + np.mean(y_train)

            else:
                self.y_tilde_train[d] = X_train @ beta_hat
                self.y_tilde_test[d] = X_test @ beta_hat

    def analyze(self):
        """
        To do here(?): 
        Find out if we can use sklearn methods, or if we have to define our own. 
        If we need our own, rewrite the next few lines with definitions of MSE and R^2.
        """
        maxdegree = self.maxdegree

        self.MSE_train = np.empty(maxdegree)
        self.MSE_test = np.empty(maxdegree)
        self.R2_train = np.empty(maxdegree)
        self.R2_test = np.empty(maxdegree)

        for d in range(maxdegree):
            self.MSE_train[d] = mean_squared_error(self.y_train[d], self.y_tilde_train[d])
            self.MSE_test[d] = mean_squared_error(self.y_test[d], self.y_tilde_test[d])

            self.R2_train[d] = r2_score(self.y_train[d], self.y_tilde_train[d])
            self.R2_test[d] = r2_score(self.y_test[d], self.y_tilde_test[d])


class RidgeRegression(Model):
    def __init__(
        self,  
        lmbda: int | list,
        x: np.ndarray | tuple[np.ndarray],
        y: np.ndarray,
        maxdegree: int,
        test_size: float = 0.2,
        scale: bool = True,
        multidim: bool = False,
        ) -> None:
        
        super().__init__(x, y, maxdegree, test_size, scale, multidim)
        if isinstance(lmbda, int):
            lmbda = [lmbda]
        self.lmbda = lmbda
        self.colors = ["forestgreen", "limegreen", "darkgoldenrod", "goldenrod", "darkorange"]
        self.name = fr"Ridge Regression ($\lambda = {self.lmbda})$"

    def predict(self):

        maxdegree = self.maxdegree

        self.y_tilde_train = {}
        self.y_tilde_test = {}
        self.beta_hat = {}
        
        for l in self.lmbda:
            self.y_tilde_train[l] = {}
            self.y_tilde_test[l] = {}
            self.beta_hat[l] = {}

            for d in range(maxdegree):
                X_train = self.X_train[d]
                X_test = self.X_test[d]
                y_train = self.y_train[d]

                self.beta_hat[d] = np.linalg.inv(X_train.T @ X_train + l*np.identity(len(X_train[0]))) @ X_train.T @ y_train
                beta_hat = self.beta_hat[d]

                # WE MUST COMMENT ON THE SCALING! REMOVE THIS WHEN DONE
                if self.scale:
                    self.y_tilde_train[l][d] = X_train @ beta_hat + np.mean(y_train)
                    self.y_tilde_test[l][d] = X_test @ beta_hat + np.mean(y_train)

                else:
                    self.y_tilde_train[l][d] = X_train @ beta_hat
                    self.y_tilde_test[l][d] = X_test @ beta_hat

    def analyze(self):
        """
        To do here(?): 
        Find out if we can use sklearn methods, or if we have to define our own. 
        If we need our own, rewrite the next few lines with definitions of MSE and R^2.
        """
        maxdegree = self.maxdegree

        self.MSE_train = {}
        self.MSE_test = {}
        self.R2_train = {}
        self.R2_test = {}

        for l in self.lmbda:
            self.MSE_train[l] = np.empty(maxdegree)
            self.MSE_test[l] = np.empty(maxdegree)
            self.R2_train[l] = np.empty(maxdegree)
            self.R2_test[l] = np.empty(maxdegree)

            for d in range(maxdegree):
                self.MSE_train[l][d] = mean_squared_error(self.y_train[d], self.y_tilde_train[l][d])
                self.MSE_test[l][d] = mean_squared_error(self.y_test[d], self.y_tilde_test[l][d])

                self.R2_train[l][d] = r2_score(self.y_train[d], self.y_tilde_train[l][d])
                self.R2_test[l][d] = r2_score(self.y_test[d], self.y_tilde_test[l][d])


class LassoRegression(Model):
    def __init__(
        self, 
        alpha: int | list,
        x: np.ndarray | tuple[np.ndarray],
        y: np.ndarray,
        maxdegree: int,
        test_size: float = 0.2,
        scale: bool = True,
        multidim: bool = False,
        ) -> None:
        
        super().__init__(x, y, maxdegree, test_size, scale, multidim)
        self.alpha = alpha
        self.colors = ["firebrick", "lightcoral", "lightseagreen", "turquoise", "blueviolet"]
        self.name = fr"Lasso Regression ($\alpha = {self.alpha})$"

    def predict(self):
        maxdegree = self.maxdegree

        self.y_tilde_train = {}
        self.y_tilde_test = {}
        self.beta_hat = {}

        for a in self.alpha:
            self.y_tilde_train[a] = {}
            self.y_tilde_test[a] = {}
            self.beta_hat[a] = {}

            for d in range(maxdegree):
                lasso = Lasso(alpha=a)
                lasso.fit(self.X_train[d], self.y_train[d])
                X_train = self.X_train[d]
                X_test = self.X_test[d]
                y_train = self.y_train[d]

                self.beta_hat[a][d] = lasso.coef_

                # WE MUST COMMENT ON THE SCALING! REMOVE THIS WHEN DONE
                if self.scale:
                    self.y_tilde_train[a][d] = lasso.predict(X_train) + np.mean(y_train)
                    self.y_tilde_test[a][d] = lasso.predict(X_test) + np.mean(y_train)

                else:
                    self.y_tilde_train[a][d] = lasso.predict(X_train)
                    self.y_tilde_test[a][d] = lasso.predict(X_test)

    def analyze(self):
        """
        To do here(?): 
        Find out if we can use sklearn methods, or if we have to define our own. 
        If we need our own, rewrite the next few lines with definitions of MSE and R^2.
        """
        maxdegree = self.maxdegree

        self.MSE_train = {}
        self.MSE_test = {}
        self.R2_train = {}
        self.R2_test = {}

        for a in self.alpha:
            self.MSE_train[a] = np.empty(maxdegree)
            self.MSE_test[a] = np.empty(maxdegree)
            self.R2_train[a] = np.empty(maxdegree)
            self.R2_test[a] = np.empty(maxdegree)

            for d in range(maxdegree):


                self.MSE_train[a][d] = mean_squared_error(self.y_train[d], self.y_tilde_train[a][d])
                self.MSE_test[a][d] = mean_squared_error(self.y_test[d], self.y_tilde_test[a][d])

                self.R2_train[a][d] = r2_score(self.y_train[d], self.y_tilde_train[a][d])
                self.R2_test[a][d] = r2_score(self.y_test[d], self.y_tilde_test[a][d])


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
        y = Franke_function(x1, x2)
        return (x1, x2), y

    else:
        x = np.linspace(-3, 3, n).reshape(-1, 1)
        y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)
        return x, y


def main():
    n = 8
    seed = 8
    test_size = 0.2
    maxdegree = 4
    scale = True
    multidim = True

    lmbda = [0.0001, 0.001, 0.01, 0.1, 1.0]
    alpha = [0.1, 0.5, 0.9, 1.5, 3.0]
    # x = np.ones(n).reshape(-1, 1)
    # y = x * 2
    # x, y = generate_data(n, seed, True)
    # OLS = OrdinaryLeastSquares(x, y, 4, scale = False, multidim=True)
    # print(OLS.design_matrices[3])
    
    fig, axs = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    Z = Franke_function(X, Y, noise=False)
    surf = axs[0].plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # plt.show()
    x, y = generate_data(n, seed, multidim)
    # x = np.ones(n).reshape(-1, 1)
    # y = x * 2
    OLS = OrdinaryLeastSquares(x, y, maxdegree, scale=False, multidim=multidim, test_size=test_size)
    OLS.predict()
    # OLS.analyze()
    # beta = OLS.beta_hat[3]
    # surf = axs[1].plot_surface(X, Y, OLS.design_matrices[3] @ beta)
    

    


    plt.show()
    # OLS.plot_train_test_and_parameters()

    # Ridge = RidgeRegression(lmbda, x, y, maxdegree, scale=scale, multidim=multidim, test_size=test_size)
    # Ridge.predict()
    # Ridge.analyze()
    # # Ridge.plot_MSE_R2_beta()

    # Lasso = LassoRegression(alpha, x, y, maxdegree, scale=scale, multidim=multidim, test_size=test_size)
    # Lasso.predict()
    # Lasso.analyze()
    # # Lasso.plot_MSE_R2_beta()


if __name__ == "__main__":
    main()
