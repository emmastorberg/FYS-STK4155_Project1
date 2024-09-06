# Importing various packages
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.model_selection import train_test_split

np.random.seed(8)
n = 100
test_size = 0.2
max_degree = 5
scaling = True

# Helper functions
def design_matrix(x, degree, scaling=scaling):
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

def predict(X_train, X_test, y_train, y_test, scaling=scaling):
    beta_hat = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

    # WE MUST COMMENT ON THE SCALING! REMOVE THIS WHEN DONE
    if scaling:
        y_tilde_train = X_train @ beta_hat + np.mean(y_train)
        y_tilde_test = X_test @ beta_hat + np.mean(y_train)

    else:
        y_tilde_train = X_train @ beta_hat
        y_tilde_test = X_test @ beta_hat

    beta = [b for sublist in beta_hat for b in sublist]

    return beta, y_tilde_train, y_tilde_test

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

train_mse = np.zeros(max_degree)
test_mse = np.zeros(max_degree)

train_r2 = np.zeros(max_degree)
test_r2 = np.zeros(max_degree)

parameters = np.zeros((2, int(max_degree*(max_degree+1)/2)))

# Helper arrays
names = [""]*int(max_degree*(max_degree+1)/2)
degrees = range(1,max_degree+1)

for degree in degrees:
    X = design_matrix(x, degree)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    beta, y_tilde_train, y_tilde_test = predict(X_train, X_test, y_train, y_test)

    """
    To do here(?): 
    Find out if we can use sklearn methods, or if we have to define our own. 
    If we need our own, rewrite the next few lines with definitions of MSE and R^2.
    """

    train_mse[degree-1] = mean_squared_error(y_train, y_tilde_train)
    test_mse[degree-1] = mean_squared_error(y_test, y_tilde_test)

    train_r2[degree-1] = r2_score(y_train, y_tilde_train)
    test_r2[degree-1] = r2_score(y_test, y_tilde_test)

    # Making parameters plottable
    prev_degree = degree - 1
    shift = int(prev_degree*(prev_degree+1)/2)

    for i in range(len(beta)):
        parameters[0][shift + i] = degree   # Makes a list incrementing up the degrees to correspond to the betas we store
        parameters[1][shift + i] = beta[i]  # Stores betas 
        names[shift + i] = fr"$\beta_{i}$"

# Plotting statistical metrics
fig, axs = plt.subplots(1, 2, figsize=(16, 9))

axs[0].plot(degrees, train_mse, label="MSE of Training Data", color="royalblue", linestyle="dotted", marker="o")
axs[0].plot(degrees, test_mse, label="MSE of Testing Data", color="cornflowerblue", linestyle="dashed", marker="o")
axs[0].plot(degrees, train_r2, label=r"$R^2$ Score of Training Data", color="chocolate", linestyle="dotted", marker="o")
axs[0].plot(degrees, test_r2, label=r"$R^2$ Score of Testing Data", color="sandybrown", linestyle="dashed", marker="o")
axs[0].set_title("Statistical Metrics")
axs[0].set_xlabel("Degree of Polynomial Model")
axs[0].xaxis.set_major_locator(MultipleLocator(1))
axs[0].set_ylabel("Value")
axs[0].legend()
axs[0].grid(True)

# Plotting optimal parameters
axs[1].scatter(parameters[0], parameters[1], color="orchid")

# This labeling looks a bit wonky. Fix later if necessary.
for degree, b, name in zip(parameters[0], parameters[1], names):
    axs[1].annotate(name+f" = {b:.3f}", (degree, b), textcoords="offset points", xytext=(60, 7), ha='right')
    
axs[1].set_title("Optimal Parameters")
axs[1].set_xlabel("Degree of Polynomial Model")
axs[1].xaxis.set_major_locator(MultipleLocator(1))
axs[1].set_ylabel("Value")
axs[1].grid(True)

#fig.suptitle(f"Title of both subplots here, if we want one")
plt.tight_layout()
plt.show()