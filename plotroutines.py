from collections.abc import Iterable

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from imageio.v2 import imread
import cmasher as cmr

import results

# plt.rcParams.update({
#     "text.usetex": True,                            # Use LaTeX for all text
#     "font.family": "serif",                         # Use a serif font
#     "font.serif": ["Computer Modern"],              # Use the Computer Modern font
#     "text.latex.preamble": r"\usepackage{amsmath}"  # If you want to include additional LaTeX packages
# })


def plot_Franke_function():
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = results.Franke_function(X, Y, noise=False)
    Z_noise = results.Franke_function(X, Y, noise=True)

    fig, ax0 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5,5))
    ax0.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=False)
    ax0.set_zlim(0, 1.05)
    ax0.xaxis.set_major_locator(MultipleLocator(0.5))
    ax0.yaxis.set_major_locator(MultipleLocator(0.5))
    ax0.zaxis.set_major_locator(MultipleLocator(0.5))
    plt.savefig("figures/Figure1a.pdf")

    fig, ax1 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5,5))
    ax1.plot_surface(X, Y, Z_noise, cmap=cm.coolwarm, antialiased=False)
    ax1.set_zlim(0, 1.05)
    ax1.xaxis.set_major_locator(MultipleLocator(0.5))
    ax1.yaxis.set_major_locator(MultipleLocator(0.5))
    ax1.zaxis.set_major_locator(MultipleLocator(0.5))
    plt.savefig("figures/Figure1b.pdf")


def plot_terrain(n: int, start: int = 0, filename="datasets/SRTM_data_Norway_1.tif"):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)

    terrain = imread(filename)
    section = terrain[start : start+n, start : start+n]

    fig, axs = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5,5))
    axs.plot_surface(X, Y, section, cmap=cm.gist_earth, label="terrain")

    axs.set_xticks([])
    axs.set_yticks([])

    axs.view_init(elev=27, azim=37, roll=0)

    plt.savefig("figures/Figure2.pdf")


def plot_mse_polydegree_OLS(res_instance_F, res_instance_T):
    # Configure Matplotlib to use LaTeX
    param = None
    degrees = res_instance_F.degrees
    mse_train_F = res_instance_F.mse_train[param]
    mse_test_F = res_instance_F.mse_test[param]
    mse_train_T = res_instance_T.mse_train[param]
    mse_test_T = res_instance_T.mse_test[param]

    sns.set_theme()
    fig, axs = plt.subplots(1, 2)

    axs[0].plot(degrees, mse_train_F, label="Training data", linestyle="solid", marker="o", color="royalblue")
    axs[0].plot(degrees, mse_test_F, label="Testing data", linestyle="dashed", marker="o", color="royalblue")
    axs[0].xaxis.set_major_locator(MultipleLocator(1))

    axs[1].plot(degrees, mse_train_T, label="Training data", linestyle="solid", marker="o", color="royalblue")
    axs[1].plot(degrees, mse_test_T, label="Testing data", linestyle="dashed", marker="o", color="royalblue")
    axs[1].xaxis.set_major_locator(MultipleLocator(1))

    axs[0].set_title(r"$\text{Franke Function}$")#, fontname="Modern")
    axs[1].set_title(r"$\text{Terrain Data}$")#, fontname="Modern")
    axs[0].legend()
    axs[1].legend()

    plt.show()
    


def plot_error_bias_variance(
    instance,
    n: int,
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
    error, bias, variance = instance.bootstrap_resampling(n=n, plot=False)
    degrees = instance.degrees
    plt.plot(degrees, error, label="Error")
    plt.plot(degrees, bias, label="Bias")
    plt.plot(degrees, variance, label="Variance")
    plt.title(f"Bias-variance tradeoff with varying complexity of polynomial model")
    if y_limit:
        plt.ylim(top=1.5)
    plt.legend()
    plt.show()


def plot_MSE_and_R2_scores(instance, params: None | Iterable = None) -> None:
    """Plots MSE and R2 scores for the model as a function 
    of polynomial degree in the same plot. One plot is 
    generated per parameter since the comparison is to 
    polynomial degree.

    Args:
        params (None | Iterable, optional): Iterable containing 
        various parameters, each of which will generate a 
        plot. Will use instance.params if nothing else is 
        specified. Defaults to None.
    """
    params = instance.params
    for param in params:
        fig, ax = plt.subplots()
        ax.plot(
            instance.degrees,
            instance.mse_train[param],
            label="MSE of Training Data",
            linestyle="dotted",
            marker="o",
        )
        ax.plot(
            instance.degrees,
            instance.mse_test[param],
            label="MSE of Testing Data",
            linestyle="dashed",
            marker="o",
        )
        ax.plot(
            instance.degrees,
            instance.r2_score_train[param],
            label=r"$R^2$ score of Training Data",
            linestyle="dotted",
            marker="o",
        )
        ax.plot(
            instance.degrees,
            instance.r2_score_test[param],
            label=r"$R^2$ Score of Testing Data",
            linestyle="dashed",
            marker="o",
        )
        ax.set_title(f"Statistical Metrics")  # of {instance.name} Method")
        ax.set_xlabel("Degree of Polynomial Model")
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()
        if instance.param_label is not None:
            fig.suptitle(f"{instance.param_label} = {param}")
        plt.show()


def plot_optimal_betas(instance, params: None | Iterable = None) -> None:
    """Plots the optimal coefficients of the model that 
    are in instance.beta_hat. If model has multiple parameters, 
    each of these will generate a separate plot.

    Args:
        params (None | Iterable, optional): Iterable containing 
        various parameters, each of which will generate a 
        plot. Will use instance.params if nothing else is 
        specified. Defaults to None.
    """
    # params = instance._get_params(params)
    params = instance.params
    for param in params:
        fig, ax = plt.subplots()
        for degree, beta in instance.beta_hat[param].items():
            ax.scatter([degree] * len(beta), beta)
            # ax.annotate(fr"\beta_{d+1}"+f" = {beta:.3f}", (d+1, beta), textcoords="offset points", xytext=(60, 7), ha='right')
        ax.set_title(f"Optimal Parameters of {instance.name} Method")
        ax.set_xlabel("Degree of Polynomial Model")
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.set_ylabel("Value")
        ax.grid(True)
        if instance.param_label is not None:
            fig.suptitle(f"{instance.param_label} = {param}")
        plt.show()


def plot_kfold_per_degree(instance, k: int, params: None | Iterable = None) -> None:
    """Plots mean squared error as a function of polynomial degree after a cross 
    validation resampling has been performed with k folds.

    Args:
        k (int): number of folds to use in K-Fold cross validation
        params (None | Iterable, optional): Iterable containing 
        various parameters, each of which will generate a 
        plot. Will use instance.params if nothing else is 
        specified. Defaults to None. Defaults to None.
    """
    params = instance._get_params(params)
    for param in params:
        fig, ax = plt.subplots()
        estimated_MSE_kfold = np.empty(instance.maxdegree)
        for degree in instance.degrees:
            estimated_MSE_kfold[degree - 1] = instance.kfold_cross_validation(
                k, degree, param
            )
        ax.plot(instance.degrees, estimated_MSE_kfold)
        ax.set_title(f"K-fold Cross Validation of {instance.name} Method")
        ax.set_xlabel("Degree of Polynomial Model")
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.set_ylabel("MSE")
        if instance.param_label is not None:
            fig.suptitle(f"{instance.param_label} = {param}")
        plt.show()


def plot_kfold_per_param(instance, k: int, degree: None | int = None, params: None | Iterable = None) -> None:
    """_summary_

    Args:
        k (int): number of folds to use in K-Fold cross validation
        degree (None | int, optional): Degree to plot MSE for. Uses instance.maxdegree if none is given. Defaults to None.
        params (None | Iterable, optional): Iterable containing various parameters. Will use instance.params if nothing else 
        is specified. Defaults to None. 
    """
    if degree is None:
        degree = instance.maxdegree
    if instance.name == "Ordinary Least Squares":
        return  # raise an error?
    fig, ax = plt.subplots()
    params = instance._get_params(params)
    estimated_MSE_kfold = np.empty(len(params))
    for i, param in enumerate(params):
        estimated_MSE_kfold[i] = instance.kfold_cross_validation(k, degree, param)
    ax.plot(np.log10(params), estimated_MSE_kfold, label="KFold")
    ax.legend()
    ax.set_title(f"K-fold Cross Validation of {instance.name} Method")
    ax.set_xlabel(instance.param_label)
    ax.set_ylabel("MSE")
    plt.show()


def plot_CV_table(instance, df, filename="CV_table.png", param_log: bool = True):
    fig, ax = plt.subplots()
    cmap = sns.color_palette("viridis", as_cmap=True)
    sns.heatmap(df.T, cmap=cmap, annot=True, fmt=".2f", cbar=True, ax=ax)
    ax.set_title(f'MSE K-fold CV of {instance.name} Method', fontsize=16)
    ax.set_xlabel("Hyper Parameters")
    ax.set_ylabel("Degree of Polynomial Model")
    plt.show()

def plot_bootstrap_per_degree(instance, num_bootstraps):
    # More code here...

    # error, bias, variance = instance.bootstrap_resampling(...)
    # instance._plot_error_bias_variance(error, bias, variance, instance.degrees, "complexity of polynomial model")
    raise NotImplementedError


def plot_bootstrap_per_datasize(instance, num_bootstraps):
    # More code here...

    # error, bias, variance = instance.bootstrap_resampling(...)
    # instance._plot_error_bias_variance(error, bias, variance, instance.degrees, "size of data set")
    raise NotImplementedError


def plot_bootstrap_per_numbootstrap(instance, num_bootstraps):
    # More code here...

    # error, bias, variance = instance.bootstrap_resampling(...)
    # instance._plot_error_bias_variance(error, bias, variance, instance.degrees, "number of bootstrap resamplings")
    raise NotImplementedError


if __name__ == "__main__":
    plot_Franke_function()
    plot_terrain(50, 700)