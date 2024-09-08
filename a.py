# --------------------------------------------- CONFIGURATION AND IMPORTS ----------------------------------------
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.model_selection import train_test_split

np.random.seed(8)   # Using a set seed for consistency in testing
n = 100             # Number of data points
test_size = 0.2     # Proportion of data used in testing
max_degree = 5      # Max degree of polynomial models plotted
scaling = True      # Use scaling as default

"""
General to-do: 
 - Add doc strings (Emma should probably be the one to do this before/on Sunday, September 8th)
 - Implement Franke function as data set and modify design matrix/code accordingly

 Part-specific to-dos are scattered throughout the code and collected as much as possible
 in Overleaf and in the README.
"""

# --------------------------------------- HELPER VARIABLES, ARRAYS AND METHODS ---------------------------------------------

degrees = range(1,max_degree+1)
lmbda_list = [0.0001, 0.001, 0.01, 0.1, 1.0]
alpha_list = [0.1, 0.5, 0.9, 1.5, 3.0]

def design_matrix(x: np.array, degree: int, scaling: bool=scaling) -> np.array:
    """
    This function will need to be changed to
    accommodate multi-variable functions.
    """

    X = np.zeros((len(x),degree))

    # WE MUST COMMENT ON THE SCALING! REMOVE THIS WHEN DONE
    if scaling:
        for i in range(degree):
            X[:,i] = x[:,0]**(i+1) 
            mean = np.mean(X[:,i])
            X[:,i] -= mean
    else:
        for i in range(degree):
            X[:,i] = x[:,0]**(i+1) 
    
    return X

def plot_train_test_and_parameters(model, train_mse, test_mse, train_r2, test_r2, parameters, point_labels, standard_colors=False):
        if standard_colors:
            colors = ["royalblue", "cornflowerblue", "chocolate", "sandybrown","orchid"]

        else:
            colors = model.colors
            
        # Plotting statistical metrics
        fig, axs = plt.subplots(1, 2, figsize=(16, 9))

        axs[0].plot(degrees, train_mse, label="MSE of Training Data", color=colors[0], linestyle="dotted", marker="o")
        axs[0].plot(degrees, test_mse, label="MSE of Testing Data", color=colors[1], linestyle="dashed", marker="o")
        axs[0].plot(degrees, train_r2, label=r"$R^2$ Score of Training Data", color=colors[2], linestyle="dotted", marker="o")
        axs[0].plot(degrees, test_r2, label=r"$R^2$ Score of Testing Data", color=colors[3], linestyle="dashed", marker="o")
        axs[0].set_title("Statistical Metrics")
        axs[0].set_xlabel("Degree of Polynomial Model")
        axs[0].xaxis.set_major_locator(MultipleLocator(1))
        axs[0].set_ylabel("Value")
        axs[0].legend()
        axs[0].grid(True)

        # Plotting optimal parameters
        axs[1].scatter(parameters[0], parameters[1], color=colors[4])

        """
        This labeling looks a bit wonky. Fix later if necessary.
        """
        for degree, b, label in zip(parameters[0], parameters[1], point_labels):
            axs[1].annotate(label+f" = {b:.3f}", (degree, b), textcoords="offset points", xytext=(60, 7), ha='right')

        axs[1].set_title("Optimal Parameters")
        axs[1].set_xlabel("Degree of Polynomial Model")
        axs[1].xaxis.set_major_locator(MultipleLocator(1))
        axs[1].set_ylabel("Value")
        axs[1].grid(True)

        fig.suptitle(model.name)
        plt.tight_layout()
        plt.show()

# --------------------------------------------- DEFINING CLASS FOR DIFFERENT MODELS ---------------------------------------

class Model:
    def __init__(self, degree: int, x: np.array, y: np.array) -> None:
        self.degree = degree
        self.x = x
        self.y = y
        self.X = design_matrix(x, degree)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, y, test_size=test_size)

    def analyze(self) -> None:
        """
        To do here(?): 
        Find out if we can use sklearn methods, or if we have to define our own. 
        If we need our own, rewrite the next few lines with definitions of MSE and R^2.
        """
        self.train_mse = mean_squared_error(self.y_train, self.y_tilde_train)
        self.test_mse = mean_squared_error(self.y_test, self.y_tilde_test)

        self.train_r2 = r2_score(self.y_train, self.y_tilde_train)
        self.test_r2 = r2_score(self.y_test, self.y_tilde_test)

    def make_plottable_parameters(self) -> None:
        return [b for sublist in self.beta_hat for b in sublist]


class OrdinaryLeastSquares(Model):
    def __init__(self, degree: int, x: np.array, y: np.array) -> None:
        super().__init__(degree, x, y)
        self.colors = ["royalblue", "cornflowerblue", "chocolate", "sandybrown","orchid"]
        self.name = "Ordinary Least Squares"

    def predict(self) -> None:
        self.beta_hat = np.linalg.inv(self.X_train.T @ self.X_train) @ self.X_train.T @ self.y_train 

        # WE MUST COMMENT ON THE SCALING! REMOVE THIS WHEN DONE
        if scaling:
            self.y_tilde_train = self.X_train @ self.beta_hat + np.mean(self.y_train)
            self.y_tilde_test = self.X_test @ self.beta_hat + np.mean(self.y_train)

        else:
            self.y_tilde_train = self.X_train @ self.beta_hat
            self.y_tilde_test = self.X_test @ self.beta_hat


class RidgeRegression(Model):
    def __init__(self, degree: int, x: np.array, y: np.array, lmbda: float) -> None:
        super().__init__(degree, x, y)
        self.lmbda = lmbda
        self.colors = ["forestgreen", "limegreen", "darkgoldenrod", "goldenrod", "darkorange"]
        self.name = fr"Ridge Regression ($\lambda = {self.lmbda})$"

    def predict(self) -> None:
        self.beta_hat = np.linalg.inv(self.X_train.T @ self.X_train + self.lmbda*np.identity(self.degree)) @ self.X_train.T @ self.y_train

        # WE MUST COMMENT ON THE SCALING! REMOVE THIS WHEN DONE
        if scaling:
            self.y_tilde_train = self.X_train @ self.beta_hat + np.mean(self.y_train)
            self.y_tilde_test = self.X_test @ self.beta_hat + np.mean(self.y_train)

        else:
            self.y_tilde_train = self.X_train @ self.beta_hat
            self.y_tilde_test = self.X_test @ self.beta_hat

class LassoRegression(Model):
    def __init__(self, degree: int, x: np.array, y: np.array, alpha: float) -> None:
        super().__init__(degree, x, y)
        self.alpha = alpha
        self.colors = ["firebrick", "lightcoral", "lightseagreen", "turquoise", "blueviolet"]
        self.name = fr"Lasso Regression ($\alpha = {self.alpha})$"

    def predict(self) -> None:
        lasso = Lasso(alpha=self.alpha)
        lasso.fit(self.X_train, self.y_train)
        self.beta_hat = lasso.coef_

        # WE MUST COMMENT ON THE SCALING! REMOVE THIS WHEN DONE
        if scaling:
            self.y_tilde_train = lasso.predict(self.X_train) + np.mean(self.y_train)
            self.y_tilde_test = lasso.predict(self.X_test) + np.mean(self.y_train)

        else:
            self.y_tilde_train = lasso.predict(self.X_train)
            self.y_tilde_test = lasso.predict(self.X_test)


# ----------------------------------------------- EXECUTABLE CODE ---------------------------------------------

# Making data set (temporary)
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)

"""
# IN FINAL PROGRAM:
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)


def FrankeFunction(x,y, noise=False):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    
    if noise:
        return term1 + term2 + term3 + term4 + np.random.normal(0, 0.1, x.shape)

    return term1 + term2 + term3 + term4


z = FrankeFunction(x, y)
"""

# Part a) – Ordinary Least Squares -------------------------------------------------------------------------------------------
ols_train_mse = np.zeros(max_degree)
ols_test_mse = np.zeros(max_degree)
ols_train_r2 = np.zeros(max_degree)
ols_test_r2 = np.zeros(max_degree)

ols_parameters = np.zeros((2, int(max_degree*(max_degree+1)/2))) 
ols_point_labels = [""]*int(max_degree*(max_degree+1)/2)  

for degree in degrees:
    ols = OrdinaryLeastSquares(degree, x, y)
    ols.predict()       # find optimal beta and use to make a prediction of the y-values
    ols.analyze()       # compute statistical metrics for training and testing data

    ols_train_mse[degree-1] = ols.train_mse
    ols_test_mse[degree-1] = ols.test_mse
    ols_train_r2[degree-1] = ols.train_r2
    ols_test_r2[degree-1] = ols.test_r2

    # Making parameters plottable
    beta = ols.make_plottable_parameters()
    prev_degree = degree - 1
    shift = int(prev_degree*(prev_degree+1)/2)

    for i in range(len(beta)):
        ols_parameters[0][shift + i] = degree   # Makes a list incrementing up the degrees to correspond to the betas we store
        ols_parameters[1][shift + i] = beta[i]  # Stores betas for easier plotting later
        ols_point_labels[shift + i] = fr"$\beta_{i}$"
    
plot_train_test_and_parameters(ols, ols_train_mse, ols_test_mse, ols_train_r2, ols_test_r2, ols_parameters, ols_point_labels)


# Part b) – Ridge Regression -------------------------------------------------------------------------------------------------
for lmbda in lmbda_list:
    ridge_train_mse = np.zeros(max_degree)
    ridge_test_mse = np.zeros(max_degree)
    ridge_train_r2 = np.zeros(max_degree)
    ridge_test_r2 = np.zeros(max_degree)

    ridge_parameters = np.zeros((2, int(max_degree*(max_degree+1)/2))) 
    ridge_point_labels = [""]*int(max_degree*(max_degree+1)/2)  

    for degree in degrees:
        ridge = RidgeRegression(degree, x, y, lmbda)
        ridge.predict()     # find optimal beta and use to make a prediction of the y-values
        ridge.analyze()     # compute statistical metrics for training and testing data

        ridge_train_mse[degree-1] = ridge.train_mse
        ridge_test_mse[degree-1] = ridge.test_mse
        ridge_train_r2[degree-1] = ridge.train_r2
        ridge_test_r2[degree-1] = ridge.test_r2

        # Making parameters plottable
        beta = ridge.make_plottable_parameters()
        prev_degree = degree - 1
        shift = int(prev_degree*(prev_degree+1)/2)

        for i in range(len(beta)):
            ridge_parameters[0][shift + i] = degree   # Makes a list incrementing up the degrees to correspond to the betas we store
            ridge_parameters[1][shift + i] = beta[i]  # Stores betas for easier plotting later
            ridge_point_labels[shift + i] = fr"$\beta_{i}$"

    plot_train_test_and_parameters(ridge, ridge_train_mse, ridge_test_mse, ridge_train_r2, ridge_test_r2, ridge_parameters, ridge_point_labels)

    """
    To-do:
    The results from these sections must be compared and analyzed. Maybe also interesting to plot wrt lambda and not
    just as a function of the polynomial degree, but we can do this later.
    """

# Part c) – Lasso Regression -------------------------------------------------------------------------------------------
"""
There is an issue here because in the plots, we see that we get negative R^2 scores, which I don't know if makes sense.
"""

for alpha in alpha_list:
    lasso_train_mse = np.zeros(max_degree)
    lasso_test_mse = np.zeros(max_degree)
    lasso_train_r2 = np.zeros(max_degree)
    lasso_test_r2 = np.zeros(max_degree)

    lasso_parameters = np.zeros((2, int(max_degree*(max_degree+1)/2))) 
    lasso_point_labels = [""]*int(max_degree*(max_degree+1)/2)  

    for degree in degrees:
        lasso = LassoRegression(degree, x, y, alpha)
        lasso.predict()     # find optimal beta and use to make a prediction of the y-values
        lasso.analyze()     # compute statistical metrics for training and testing data

        lasso_train_mse[degree-1] = lasso.train_mse
        lasso_test_mse[degree-1] = lasso.test_mse
        lasso_train_r2[degree-1] = lasso.train_r2
        lasso_test_r2[degree-1] = lasso.test_r2

        # Making parameters plottable
        beta = lasso.beta_hat
        prev_degree = degree - 1
        shift = int(prev_degree*(prev_degree+1)/2)

        for i in range(len(beta)):
            lasso_parameters[0][shift + i] = degree   # Makes a list incrementing up the degrees to correspond to the betas we store
            lasso_parameters[1][shift + i] = beta[i]  # Stores betas for easier plotting later
            lasso_point_labels[shift + i] = fr"$\beta_{i}$"

    plot_train_test_and_parameters(lasso, lasso_train_mse, lasso_test_mse, lasso_train_r2, lasso_test_r2, lasso_parameters, lasso_point_labels)