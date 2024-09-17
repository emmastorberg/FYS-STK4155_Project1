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
        self.y = y.flatten()
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
            X1, X2 = self.x
            X1 = X1.flatten()
            X2 = X2.flatten()

            n = len(X1)
            p = int((degree + 1) * (degree + 2) / 2 - 1)  # Number of terms in the polynomial expansion

            X = np.zeros((n, p))

            for i in range(degree):
                base_index = int((i + 1) * i / 2 + i)  # Calculate base index for the current degree

                for j in range(i + 2):
                    # Fill the design matrix with x and y raised to appropriate powers
                    X[:, base_index + j] = X1**(i + 1 - j) * X2**(j)
                    if self.scale:
                        X[:, base_index + 1] -= np.mean(X[:, base_index + 1])
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
        
    def plot_MSE_and_R2(self, ax, param = None):
        degrees = np.arange(1, self.maxdegree + 1)
        c1, c2, c3, c4, _ = self.colors
        if param is None:
            MSE_train, MSE_test, R2_train, R2_test = self.MSE_train, self.MSE_test, self.R2_train, self.R2_test
        else:
            MSE_train, MSE_test, R2_train, R2_test = self.MSE_train[param], self.MSE_test[param], self.R2_train[param], self.R2_test[param]
        ax.plot(degrees, MSE_train, label="MSE of Training Data", color=c1, linestyle="dotted", marker="o")
        ax.plot(degrees, MSE_test, label="MSE of Testing Data", color=c2, linestyle="dashed", marker="o")
        ax.plot(degrees, R2_train, label=r"$R^2$ score of Training Data", color=c3, linestyle="dotted", marker="o")
        ax.plot(degrees, R2_test, label=r"$R^2$ Score of Testing Data", color=c4, linestyle="dashed", marker="o")
        ax.set_title(f"Statistical Metrics of {self.name} Method")
        ax.set_xlabel("Degree of Polynomial Model")
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()

    def plot_optimal_beta(self, ax, param = None):
        color = self.colors[-1]
        if param is None:
            beta_hat = self.beta_hat
        else:
            beta_hat = self.beta_hat[param]
        for d, beta in beta_hat.items():
            ax.scatter([d+1] * len(beta), beta, color=color)
            # ax.annotate(fr"\beta_{d+1}"+f" = {beta:.3f}", (d+1, beta), textcoords="offset points", xytext=(60, 7), ha='right')
        ax.set_title(f"Optimal Parameters of {self.name} Method")
        ax.set_xlabel("Degree of Polynomial Model")
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.set_ylabel("Value")
        ax.grid(True)

        


    # def plot_train_test_and_parameters(self, param = None):
    #         colors = ["royalblue", "cornflowerblue", "chocolate", "sandybrown","orchid"]
                
    #         # Plotting statistical metrics
    #         fig, axs = plt.subplots(1, 2, figsize=(16, 9))
    #         degrees = np.arange(1, self.maxdegree + 1)

    #         if param is None:
    #             MSE_train = self.MSE_train
    #             MSE_test = self.MSE_test
    #             R2_train = self.R2_train
    #             R2_test = self.R2_test
    #             beta_hat = self.beta_hat
    #         else:
    #             MSE_train = self.MSE_train[param]
    #             MSE_test = self.MSE_test[param]
    #             R2_train = self.R2_train[param]
    #             R2_test = self.R2_test[param]
    #             beta_hat = self.beta_hat[param]

    #         axs[0].plot(degrees, MSE_train, label="MSE of Training Data", color=colors[0], linestyle="dotted", marker="o")
    #         axs[0].plot(degrees, MSE_test, label="MSE of Testing Data", color=colors[1], linestyle="dashed", marker="o")
    #         axs[0].plot(degrees, R2_train, label=r"$R^2$ Score of Training Data", color=colors[2], linestyle="dotted", marker="o")
    #         axs[0].plot(degrees, R2_test, label=r"$R^2$ Score of Testing Data", color=colors[3], linestyle="dashed", marker="o")
    #         axs[0].set_title("Statistical Metrics")
    #         axs[0].set_xlabel("Degree of Polynomial Model")
    #         axs[0].xaxis.set_major_locator(MultipleLocator(1))
    #         axs[0].set_ylabel("Value")
    #         axs[0].legend()
    #         axs[0].grid(True)

    #         # Plotting optimal parameters
    #         # axs[1].scatter(degrees, beta_hat, color=colors[4])

    #         # """
    #         # This labeling looks a bit wonky. Fix later if necessary.
    #         # """
    #         # for degree, b in zip(degrees, beta_hat):
    #         #     axs[1].annotate(fr"\beta_{degree}"+f" = {b:.3f}", (degree, b), textcoords="offset points", xytext=(60, 7), ha='right')

    #         axs[1].set_title("Optimal Parameters")
    #         axs[1].set_xlabel("Degree of Polynomial Model")
    #         axs[1].xaxis.set_major_locator(MultipleLocator(1))
    #         axs[1].set_ylabel("Value")
    #         axs[1].grid(True)

    #         fig.suptitle(self.name)
    #         plt.tight_layout()
    #         plt.show()


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

    def plot_MSE_and_R2(self):
        fig, ax = plt.subplots()
        super().plot_MSE_and_R2(ax)
        plt.show()

    def plot_optimal_beta(self):
        fig, ax = plt.subplots()
        super().plot_optimal_beta(ax)
        plt.show()


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
        self.name = "Ridge Regression"

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

                self.beta_hat[l][d] = np.linalg.inv(X_train.T @ X_train + l*1000*np.eye(len(X_train[0]))) @ X_train.T @ y_train
                beta_hat = self.beta_hat[l][d]

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
            # print(f"lambda = {l}: \n {self.MSE_train[l]} \n")

    def plot_MSE_and_R2(self):
        for l in self.lmbda:
            fig, ax = plt.subplots()
            super().plot_MSE_and_R2(ax, l)
            fig.suptitle(rf"$\lambda$ = {l}")
            plt.show()

    def plot_optimal_beta(self):
        for l in self.lmbda:
            fig, ax = plt.subplots()
            super().plot_optimal_beta(ax, l)
            fig.suptitle(rf"$\lambda$ = {l}")
            plt.show()

    def MSE_per_lmbda(self, degree):
        """degree must be smaller than or equal maxdegree"""
        fig, ax = plt.subplots()
        MSE_test = [self.MSE_test[l][degree-1] for l in self.lmbda]
        MSE_train = [self.MSE_train[l][degree-1] for l in self.lmbda]
        ax.plot(self.lmbda, MSE_test, linestyle="dashed", label="test")
        ax.plot(self.lmbda, MSE_train, label="train")
        ax.legend()
        ax.set_xscale("log")
        plt.show()


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
        self.name = "Lasso Regression"

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

    def plot_MSE_and_R2(self):
        for a in self.alpha:
            fig, ax = plt.subplots()
            super().plot_MSE_and_R2(ax, a)
            fig.suptitle(rf"$\alpha$ = {a}")
            plt.show()

    def plot_optimal_beta(self):
        for a in self.alpha:
            fig, ax = plt.subplots()
            super().plot_optimal_beta(ax, a)
            fig.suptitle(rf"$\alpha$ = {a}")
            plt.show()

    def MSE_per_alpha(self, degree):
        """degree must be smaller than or equal maxdegree"""
        fig, ax = plt.subplots()
        MSE_test = [self.MSE_test[a][degree-1] for a in self.alpha]
        MSE_train = [self.MSE_train[a][degree-1] for a in self.alpha]
        ax.plot(self.alpha, MSE_test, linestyle="dashed", label="train")
        ax.plot(self.alpha, MSE_train, label="test")
        ax.legend()
        plt.show()



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


def plot_true_vs_pred_Franke(n, degree, seed):
    fig, axs = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    Z = Franke_function(X, Y, noise=False)
    axs[0].plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    x, y = generate_data(n, seed, multidim=True)
    OLS = OrdinaryLeastSquares(x, y, degree, multidim=True)
    OLS.predict()
    beta = OLS.beta_hat[degree - 1]
    X_matrix = OLS.design_matrices[degree - 1]
    axs[1].plot_surface(X, Y, (X_matrix @ beta).reshape(n,n), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()



def main():
    n = 100
    seed = 8
    test_size = 0.2
    maxdegree = 10
    scale = True
    multidim = True

    lmbda = [0, 0.0001, 0.001, 0.01, 0.1, 1.0]
    alpha = [0.1, 0.5, 0.9, 1.5, 3.0]

    x, y = generate_data(n, seed, multidim)

    OLS = OrdinaryLeastSquares(x, y, maxdegree, test_size, scale, multidim)
    OLS.predict()
    OLS.analyze()
    # OLS.plot_MSE_and_R2()
    OLS.plot_optimal_beta()

    Ridge = RidgeRegression(lmbda, x, y, maxdegree, scale=scale, multidim=multidim, test_size=test_size)
    Ridge.predict()
    Ridge.analyze()
    # Ridge.plot_MSE_and_R2()
    Ridge.plot_optimal_beta()
    Ridge.MSE_per_lmbda(10)

    Lasso = LassoRegression(alpha, x, y, maxdegree, scale=scale, multidim=multidim, test_size=test_size)
    Lasso.predict()
    Lasso.analyze()
    # Lasso.plot_MSE_and_R2()
    Lasso.plot_optimal_beta()
    Lasso.MSE_per_alpha(10)


if __name__ == "__main__":
    main()
