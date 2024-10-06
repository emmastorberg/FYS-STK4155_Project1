from collections.abc import Iterable

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from imageio.v2 import imread

import results


def aesthetic_2D():
    plt.rcParams.update({
        # "text.usetex": True,                            # Use LaTeX for all text
        # "font.family": "serif",                         # Use a serif font
        # "font.serif": ["Computer Modern"],              # Use the Computer Modern font
        # "text.latex.preamble": r"\usepackage{amsmath}", # If you want to include additional LaTeX packages

        # Matplotlib style settings similar to seaborn's default style
        "axes.facecolor": "#eaeaf2",      # Background color of the plot
        "axes.edgecolor": "white",       # Color of the plot edge
        "axes.grid": True,              # Enable the grid
        "grid.color": "white",          # Grid color
        "grid.linestyle": "-",          # Grid line style
        "grid.linewidth": 1,          # Grid line width
        "axes.axisbelow": True,         # Draw gridlines below other plot elements
        "xtick.color": "gray",          # X-tick color
        "ytick.color": "gray",          # Y-tick color

        # Additional stylistic settings
        "figure.facecolor": "white",    # Background color of the figure
        "legend.frameon": True,         # Legend frame
        "legend.framealpha": 0.8,       # Legend frame transparency
        "legend.fancybox": True,        # Rounded box for the legend
        "legend.edgecolor": 'lightgray'      # Legend edge color
    })


def plot_Franke_function(n, noise: bool, ax=None, filename=""):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    Z = results.Franke_function(X, Y, noise=noise)
    
    savefig = False
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5,5))
        savefig = True

    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=False)
    ax.set_zlim(0, 1.05)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.zaxis.set_major_locator(MultipleLocator(0.5))

    if savefig:
        plt.savefig(f"figures/{filename}")
        plt.show()
        plt.close()


def plot_terrain(n: int, start: int, step: int, ax=None, terrain_file="datasets/SRTM_data_Norway_1.tif", filename=""):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)

    terrain = imread(terrain_file)
    section = terrain[start : start+(n*step) : step, start : start+(n*step) : step]

    savefig = False
    if ax is None:
        savefig = True
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5,5))
    ax.plot_surface(X, Y, section, cmap=cm.gist_earth, label="terrain")

    ax.set_xticks([])
    ax.set_yticks([])

    ax.view_init(elev=27, azim=37, roll=0)

    if savefig:
        plt.savefig(f"figures/{filename}")
        plt.show()
        plt.close()


def plot_mse_per_polydegree_OLS(res_instance_F, res_instance_T, filename):
    # Configure Matplotlib to use LaTeX
    aesthetic_2D()

    param = None
    degrees = res_instance_F.degrees
    mse_train_F = res_instance_F.mse_train[param]
    mse_test_F = res_instance_F.mse_test[param]
    mse_train_T = res_instance_T.mse_train[param]
    mse_test_T = res_instance_T.mse_test[param]

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))

    axs[0].plot(degrees, mse_train_F, label="OLS Train", linestyle="solid", marker="o", color="royalblue")
    axs[0].plot(degrees, mse_test_F, label="OLS Test", linestyle="dashed", marker="o", color="royalblue")

    axs[1].plot(degrees, mse_train_T, label="OLS Train", linestyle="solid", marker="o", color="royalblue")
    axs[1].plot(degrees, mse_test_T, label="OLS Test", linestyle="dashed", marker="o", color="royalblue")

    axs[0].set_xlabel("Degree of Polynomial Model")
    axs[1].set_xlabel("Degree of Polynomial Model")
    axs[0].set_ylabel("Mean Squared Error")
    axs[0].xaxis.set_major_locator(MultipleLocator(1))
    axs[1].xaxis.set_major_locator(MultipleLocator(1))

    axs[0].set_title("Franke Function")
    axs[1].set_title("Terrain Data")
    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(f"figures/{filename}")
    plt.show()
    plt.close()


def plot_r2_score_per_polydegree_OLS(res_instance_F, res_instance_T, filename):
    # Configure Matplotlib to use LaTeX
    aesthetic_2D()

    param = None
    degrees = res_instance_F.degrees
    r2_score_train_F = res_instance_F.r2_score_train[param]
    r2_score_test_F = res_instance_F.r2_score_test[param]
    r2_score_train_T = res_instance_T.r2_score_train[param]
    r2_score_test_T = res_instance_T.r2_score_test[param]

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))

    axs[0].plot(degrees, r2_score_train_F, label="OLS Train", linestyle="solid", marker="o", color="royalblue")
    axs[0].plot(degrees, r2_score_test_F, label="OLS Test", linestyle="dashed", marker="o", color="royalblue")

    axs[1].plot(degrees, r2_score_train_T, label="OLS Train", linestyle="solid", marker="o", color="royalblue")
    axs[1].plot(degrees, r2_score_test_T, label="OLS Test", linestyle="dashed", marker="o", color="royalblue")

    axs[0].set_xlabel("Degree of Polynomial Model")
    axs[1].set_xlabel("Degree of Polynomial Model")
    axs[0].set_ylabel(r"$R^2$ Score")
    axs[0].xaxis.set_major_locator(MultipleLocator(1))
    axs[1].xaxis.set_major_locator(MultipleLocator(1))

    axs[0].set_title("Franke Function")
    axs[1].set_title("Terrain Data")
    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(f"figures/{filename}")
    plt.show()
    plt.close()


def plot_mse_per_hyper_param_all_instances(
        res_instances_F: list, 
        res_instances_T: list, 
        degree_F: int, 
        degree_T: int,
        filename: str,
    ):
    """
    Order of instances: OLS, ridge, Lasso
    """
    aesthetic_2D()

    params = res_instances_F[1].params
    
    ols_F, ridge_F, lasso_F = res_instances_F
    ols_T, ridge_T, lasso_T = res_instances_T

    mse_train_F = np.zeros(len(params) * 3)
    mse_test_F = np.zeros(len(params) * 3)
    mse_train_T = np.zeros(len(params) * 3)
    mse_test_T = np.zeros(len(params) * 3)

    mse_train_F[::3] = ols_F.mse_train[None][degree_F-1]
    mse_test_F[::3] = ols_F.mse_test[None][degree_F-1]
    mse_train_T[::3] = ols_T.mse_train[None][degree_T-1]
    mse_test_T[::3] = ols_T.mse_test[None][degree_T-1]

    mse_train_F[1::3] = [ridge_F.mse_train[p][degree_F-1] for p in params]
    mse_test_F[1::3] = [ridge_F.mse_test[p][degree_F-1] for p in params]
    mse_train_T[1::3] = [ridge_T.mse_train[p][degree_T-1] for p in params]
    mse_test_T[1::3] = [ridge_T.mse_test[p][degree_T-1] for p in params]

    mse_train_F[2::3] = [lasso_F.mse_train[p][degree_F-1] for p in params]
    mse_test_F[2::3] = [lasso_F.mse_test[p][degree_F-1] for p in params]
    mse_train_T[2::3] = [lasso_T.mse_train[p][degree_T-1] for p in params]
    mse_test_T[2::3] = [lasso_T.mse_test[p][degree_T-1] for p in params]

    
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))

    train_labels = ["OLS Train", "Ridge Train", "LASSO Train"]
    test_labels = ["OLS Test", "Ridge Test", "LASSO Train"]
    colors = ["royalblue", "indianred", "limegreen"]

    for i in range(3):
        axs[0].plot(params, mse_train_F[i::3], label=train_labels[i], color=colors[i], linestyle="solid")
        axs[0].plot(params, mse_test_F[i::3], label=test_labels[i], color=colors[i], linestyle="dashed")
        axs[1].plot(params, mse_train_T[i::3], label=train_labels[i], color=colors[i], linestyle="solid")
        axs[1].plot(params, mse_test_T[i::3], label=test_labels[i], color=colors[i], linestyle="dashed")

    axs[0].set_xlabel(r"$\lambda$")
    axs[1].set_xlabel(r"$\lambda$")
    axs[0].set_ylabel("Mean Squared Error")
    axs[0].set_xscale("log")
    axs[1].set_xscale("log")

    axs[0].set_title("Franke Function")
    axs[1].set_title("Terrain Data")
    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()

    plt.show()
    plt.savefig(f"figures/{filename}")
    plt.close()


def plot_r2_score_per_hyper_param_all_instances(
        res_instances_F: list, 
        res_instances_T: list, 
        degree_F: int, 
        degree_T: int,
        filename,
    ):
    """
    Order of instances: OLS, Ridge, Lasso
    """
    aesthetic_2D()

    params = res_instances_F[1].params
    
    ols_F, ridge_F, lasso_F = res_instances_F
    ols_T, ridge_T, lasso_T = res_instances_T

    mse_train_F = np.zeros(len(params) * 3)
    mse_test_F = np.zeros(len(params) * 3)
    mse_train_T = np.zeros(len(params) * 3)
    mse_test_T = np.zeros(len(params) * 3)

    mse_train_F[::3] = ols_F.r2_score_train[None][degree_F-1]
    mse_test_F[::3] = ols_F.r2_score_test[None][degree_F-1]
    mse_train_T[::3] = ols_T.r2_score_train[None][degree_T-1]
    mse_test_T[::3] = ols_T.r2_score_test[None][degree_T-1]

    mse_train_F[1::3] = [ridge_F.r2_score_train[p][degree_F-1] for p in params]
    mse_test_F[1::3] = [ridge_F.r2_score_test[p][degree_F-1] for p in params]
    mse_train_T[1::3] = [ridge_T.r2_score_train[p][degree_T-1] for p in params]
    mse_test_T[1::3] = [ridge_T.r2_score_test[p][degree_T-1] for p in params]

    mse_train_F[2::3] = [lasso_F.r2_score_train[p][degree_F-1] for p in params]
    mse_test_F[2::3] = [lasso_F.r2_score_test[p][degree_F-1] for p in params]
    mse_train_T[2::3] = [lasso_T.r2_score_train[p][degree_T-1] for p in params]
    mse_test_T[2::3] = [lasso_T.r2_score_test[p][degree_T-1] for p in params]

    
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))

    train_labels = ["OLS Train", "Ridge Train", "LASSO Train"]
    test_labels = ["OLS Test", "Ridge Test", "LASSO Train"]
    colors = ["royalblue", "indianred", "limegreen"]

    for i in range(3):
        axs[0].plot(params, mse_train_F[i::3], label=train_labels[i], color=colors[i], linestyle="solid")
        axs[0].plot(params, mse_test_F[i::3], label=test_labels[i], color=colors[i], linestyle="dashed")
        axs[1].plot(params, mse_train_T[i::3], label=train_labels[i], color=colors[i], linestyle="solid")
        axs[1].plot(params, mse_test_T[i::3], label=test_labels[i], color=colors[i], linestyle="dashed")

    axs[0].set_xlabel(r"$\lambda$")
    axs[1].set_xlabel(r"$\lambda$")
    axs[0].set_ylabel(r"$R^2$ Score")
    axs[0].set_xscale("log")
    axs[1].set_xscale("log")

    axs[0].set_title(f"Franke Function (degree {degree_F})")
    axs[1].set_title(f"Terrain Data (degree {degree_T})")
    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()

    plt.show()
    plt.savefig(f"figures/{filename}")
    plt.close()


def plot_CV_table(instance, datatype, filename):
    aesthetic_2D()

    df = instance.grid_search()

    fig, ax = plt.subplots(figsize=(10.3, 6.2))
    cmap = sns.color_palette("viridis", as_cmap=True)
    sns.heatmap(df.T, cmap=cmap, annot=True, fmt=".2f", cbar=True, ax=ax)

    ax.set_title(f"{datatype} ({instance.name})", fontsize=16)
    # ax.set_yscale("log")
    ax.set_xlabel("Degree of Polynomial Model")
    ax.set_ylabel(r"$\lambda$")

    plt.tight_layout()
    plt.savefig(f"figures/{filename}")
    plt.show()
    plt.close()


def plot_mse_per_polydegree_all_instances(instances_F, instances_T, param_F, param_T, filename):
    aesthetic_2D()

    params_F = [None, param_F, param_F]
    params_T = [None, param_T, param_T]

    degrees = instances_F[0].degrees

    train_labels = ["OLS Train", "Ridge Train", "LASSO Train"]
    test_labels = ["OLS Test", "Ridge Test", "LASSO Train"]
    colors = ["royalblue", "indianred", "limegreen"]

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))

    for i in range(3):
        c = colors[i]
        axs[0].plot(degrees, instances_F[i].mse_train[params_F[i]], label=train_labels[i], c=c, linestyle="solid", marker="o")
        axs[0].plot(degrees, instances_F[i].mse_test[params_F[i]], label=test_labels[i], c=c, linestyle="dashed", marker="o")
        axs[1].plot(degrees, instances_T[i].mse_train[params_T[i]], label=train_labels[i], c=c, linestyle="solid", marker="o")
        axs[1].plot(degrees, instances_T[i].mse_test[params_F[i]], label=test_labels[i], c=c, linestyle="dashed", marker="o")
    
    axs[0].set_xlabel("Degree of Polynomial Model")
    axs[1].set_xlabel("Degree of Polynomial Model")
    axs[0].set_ylabel("Mean Squared Error")
    axs[0].xaxis.set_major_locator(MultipleLocator(1))
    axs[1].xaxis.set_major_locator(MultipleLocator(1))

    axs[0].set_title(fr"Franke Function ($\lambda$ = {param_F:.1e})")
    axs[1].set_title(fr"Terrain Data ($\lambda$ = {param_T:.1e})")
    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(f"figures/{filename}")
    plt.show()
    plt.close()


def plot_r2_score_per_polydegree_all_instances(instances_F, instances_T, param_F, param_T, filename):
    aesthetic_2D()

    params_F = [None, param_F, param_F]
    params_T = [None, param_T, param_T]

    degrees = instances_F[0].degrees

    train_labels = ["OLS Train", "Ridge Train", "LASSO Train"]
    test_labels = ["OLS Test", "Ridge Test", "LASSO Train"]
    colors = ["royalblue", "indianred", "limegreen"]

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))

    for i in range(3):
        c = colors[i]
        axs[0].plot(degrees, instances_F[i].r2_score_train[params_F[i]], label=train_labels[i], c=c, linestyle="solid", marker="o")
        axs[0].plot(degrees, instances_F[i].r2_score_test[params_F[i]], label=test_labels[i], c=c, linestyle="dashed", marker="o")
        axs[1].plot(degrees, instances_T[i].r2_score_train[params_T[i]], label=train_labels[i], c=c, linestyle="solid", marker="o")
        axs[1].plot(degrees, instances_T[i].r2_score_test[params_F[i]], label=test_labels[i], c=c, linestyle="dashed", marker="o")
    
    axs[0].set_xlabel("Degree of Polynomial Model")
    axs[1].set_xlabel("Degree of Polynomial Model")
    axs[0].set_ylabel(r"$R^2$ Score")
    axs[0].xaxis.set_major_locator(MultipleLocator(1))
    axs[1].xaxis.set_major_locator(MultipleLocator(1))

    axs[0].set_title(fr"Franke Function ($\lambda$ = {param_F:.1e})")
    axs[1].set_title(fr"Terrain Data ($\lambda$ = {param_T:.1e})")
    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(f"figures/{filename}")
    plt.show()
    plt.close()
    

def plot_optimal_coefficients(instances_F, instances_T, param_F, param_T, filename):
    aesthetic_2D()

    params_F = [None, param_F, param_F]
    params_T = [None, param_T, param_T]

    degrees = instances_F[0].degrees

    colors = ["royalblue", "indianred", "limegreen"]

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))

    for i in range(3):
        c = colors[i]
        for degree in degrees:
            beta_hat_F = instances_F[i].beta_hat[params_F[i]][degree]
            beta_hat_T = instances_T[i].beta_hat[params_T[i]][degree]
            axs[0].scatter([degree] * len(beta_hat_F), beta_hat_F, c=c, marker="x")
            axs[1].scatter([degree] * len(beta_hat_T), beta_hat_T, c=c, marker="x")
    
    axs[0].set_xlabel("Degree of Polynomial Model")
    axs[1].set_xlabel("Degree of Polynomial Model")
    axs[0].set_ylabel("Value")
    axs[0].xaxis.set_major_locator(MultipleLocator(1))
    axs[1].xaxis.set_major_locator(MultipleLocator(1))

    axs[0].set_title(fr"Franke Function ($\lambda$ = {param_F:.1e})")
    axs[1].set_title(fr"Terrain Data ($\lambda$ = {param_T:.1e})")

    plt.tight_layout()
    plt.savefig(f"figures/{filename}")
    plt.show()
    plt.close()


def plot_error_bias_variance(
    instance_F,
    instance_T,
    n: int,
    filename,
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
    aesthetic_2D()

    error_F, bias_F, variance_F = instance_F.bootstrap_resampling(n)
    error_T, bias_T, variance_T = instance_T.bootstrap_resampling(n)
    degrees = instance_F.degrees

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))

    colors = ["darkblue", "deepskyblue", "cornflowerblue"]

    axs[0].plot(degrees, error_F, label="Error", color=colors[0], marker="o")
    axs[0].plot(degrees, bias_F, label="Bias", color=colors[1], marker="o")
    axs[0].plot(degrees, variance_F, label="Variance", color=colors[2], marker="o")

    axs[1].plot(degrees, error_T, label="Error", color=colors[0], marker="o")
    axs[1].plot(degrees, bias_T, label="Bias", color=colors[1], marker="o")
    axs[1].plot(degrees, variance_T, label="Variance", color=colors[2], marker="o")

    axs[0].set_xlabel("Degree of Polynomial Model")
    axs[1].set_xlabel("Degree of Polynomial Model")
    axs[0].set_ylabel("Value")
    axs[0].xaxis.set_major_locator(MultipleLocator(1))
    axs[1].xaxis.set_major_locator(MultipleLocator(1))

    axs[0].set_title(fr"Franke Function")
    axs[1].set_title(fr"Terrain Data")
    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(f"figures/{filename}")
    plt.show()
    plt.close()


def plot_kfold_per_degree(
        instances_F, 
        instances_T,
        kfolds: int, 
        degree: int, 
        params_F,
        params_T,
        bootstrap_error_F,
        bootstrap_error_T,
        filename,
    ) -> None:
    """Plots mean squared error as a function of polynomial degree after a cross 
    validation resampling has been performed with k folds.

    Args:
        k (int): number of folds to use in K-Fold cross validation
        params (None | Iterable, optional): Iterable containing 
        various parameters, each of which will generate a 
        plot. Will use instance.params if nothing else is 
        specified. Defaults to None. Defaults to None.
    """
    aesthetic_2D()

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))

    estimated_mse_kfold_F = np.empty(len(kfolds) * 3)
    estimated_mse_kfold_T = np.empty(len(kfolds) * 3)

    for i, fold in enumerate(kfolds):
        for j in range(3):
            estimated_mse_kfold_F[(i * 3) + j] = instances_F[j].kfold_CV(
                fold, degree, params_F[j]
            )
            estimated_mse_kfold_T[(i * 3) + j] = instances_T[j].kfold_CV(
                fold, degree, params_T[j]
            )

    colors = ["royalblue", "indianred", "limegreen"]
    labels = ["CV OLS", "CV Ridge", "CV LASSO"]

    for i in range(3):
        axs[0].plot(kfolds, estimated_mse_kfold_F[i::3], label=labels[i], color=colors[i], marker="o")
        axs[1].plot(kfolds, estimated_mse_kfold_T[i::3], label=labels[i], color=colors[i], marker="o")

    axs[0].plot(kfolds, [bootstrap_error_F] * len(kfolds), label="Bootstrap OLS", color="darkblue", linestyle="dashed")
    axs[1].plot(kfolds, [bootstrap_error_T] * len(kfolds), label="Bootstrap OLS", color="darkblue", linestyle="dashed")

    axs[0].set_xlabel("Folds")
    axs[1].set_xlabel("Folds")
    axs[0].set_ylabel("Value")
    axs[0].xaxis.set_major_locator(MultipleLocator(1))
    axs[1].xaxis.set_major_locator(MultipleLocator(1))

    axs[0].set_title(fr"Franke Function")
    axs[1].set_title(fr"Terrain Data")
    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(f"figures/{filename}")
    plt.show()
    plt.close()


def plot_true_vs_predicted_Franke(y_tilde_high_n, y_tilde_low_n, high_n, low_n, filename):
    fig, axs = plt.subplots(2, 2, subplot_kw={"projection": "3d"}, figsize=(8, 8))

    plot_Franke_function(high_n, noise=True, ax=axs[0,0])
    plot_Franke_function(low_n, noise=True, ax=axs[1,0])

    x1_high_n = np.linspace(0, 1, high_n)
    x2_high_n = np.linspace(0, 1, high_n)
    x1_low_n = np.linspace(0, 1, low_n)
    x2_low_n = np.linspace(0, 1, low_n)

    X1_high, X2_high = np.meshgrid(x1_high_n, x2_high_n)
    X1_low, X2_low = np.meshgrid(x1_low_n, x2_low_n)

    Y_tilde_high = y_tilde_high_n.reshape(high_n, high_n)
    Y_tilde_low = y_tilde_low_n.reshape(low_n, low_n)

    axs[0,1].plot_surface(X1_high, X2_high, Y_tilde_high)
    axs[1,1].plot_surface(X1_low, X2_low, Y_tilde_low)

    axs[0,1].set_zlim(0, 1.05)
    axs[1,1].set_zlim(0, 1.05)
    axs[0,1].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,1].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,1].zaxis.set_major_locator(MultipleLocator(0.5))
    axs[1,1].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[1,1].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[1,1].zaxis.set_major_locator(MultipleLocator(0.5))

    plt.show()
    # plt.savefig(f"figures/{filename}")
    plt.close()

def plot_true_vs_predicted_terrain(
        y_tilde_high_n,
        y_tilde_low_n,
        high_n,
        low_n,
        start,
        step_high_n,
        step_low_n,
        terrain_file,
        filename,
    ):
    fig, axs = plt.subplots(2, 2, subplot_kw={"projection": "3d"}, figsize=(8, 8))

    plot_terrain(high_n, start, step_high_n, axs[0,0], terrain_file)
    plot_terrain(low_n, start, step_low_n, axs[1,0], terrain_file)

    x1_high_n = np.linspace(0, 1, high_n)
    x2_high_n = np.linspace(0, 1, high_n)
    x1_low_n = np.linspace(0, 1, low_n)
    x2_low_n = np.linspace(0, 1, low_n)

    X1_high, X2_high = np.meshgrid(x1_high_n, x2_high_n)
    X1_low, X2_low = np.meshgrid(x1_low_n, x2_low_n)

    Y_tilde_high = y_tilde_high_n.reshape(high_n, high_n)
    Y_tilde_low = y_tilde_low_n.reshape(low_n, low_n)

    axs[0,1].plot_surface(X1_high, X2_high, Y_tilde_high)
    axs[1,1].plot_surface(X1_low, X2_low, Y_tilde_low)

    axs[0,1].set_xticks([])
    axs[1,1].set_yticks([])

    axs[0,1].view_init(elev=27, azim=37, roll=0)
    axs[1,1].view_init(elev=27, azim=37, roll=0)

    plt.show()
    # plt.savefig(f"figures/{filename}")
    plt.close()
