import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from scipy.spatial.distance import cdist
from scipy.linalg import cho_solve

from test_functions import *

# 1D test problem function
def tf(x):
    y = x**2 - x * np.log(x + 1.) + (x - 0.5) * np.sin(2. * np.pi * x)
    return y

def d_tf(x):
    dy = 2. * x - np.log(x + 1.) - x / (x + 1.) + np.sin(2. * np.pi * x) \
        + (x - 0.5) * 2. * np.pi * np.cos(2. * np.pi * x)
    return dy

def get_1D_problem_data():
    xs = np.linspace(0.1, 0.9, 9).reshape(-1, 1)
    ys = tf(xs)
    ps = np.linspace(0, 1, 201).reshape(-1, 1)
    return xs, ys, ps


def get_nD_problem_data(ndims, ranges, npoints, fun):
    ranges = np.atleast_2d(ranges)
    train_data = np.zeros((npoints, ndims))
    train_targets = np.zeros((npoints, 1))
    for dim in range(ndims):
        start_val = ranges[dim][0]
        stop_val = ranges[dim][1]
        train_data[:,dim] = np.linspace(start_val, stop_val, npoints)
    for point in range(npoints):
        train_targets[point] = fun(train_data[point, :])
    return train_data, train_targets


def partial_skl_kernel(ndim, kernel_type):
    init = (1.0,)
    ls_bounds = (1e-2, 1e2)
    if kernel_type == "SE":
        return RBF(init * ndim, length_scale_bounds=ls_bounds)
    else:
        raise NotImplementedError("gradient for kernel type not implemented")

def compute_gp_grad_mu(skl_gp, input_scaler, output_scaler,
        alpha_vec, xs_t, ps_t):
    ndims = xs_t.shape[1]
    pred_cov = skl_gp.kernel_(ps_t, xs_t)
    length_scale_hyperparams = np.exp(skl_gp.kernel_.theta[1:]) # [scaling, length scale params] - sklearn stores the log of theta
    pred_deriv = compute_pred_cov_deriv(ps_t, xs_t, pred_cov,
        length_scale_hyperparams)
    # scaled gradient
    grad_mu = np.einsum("ijk, j -> ik", pred_deriv, alpha_vec)
    for kk in range(ndims):
        scale_factor = output_scaler.scale_[kk] / input_scaler.scale_[kk]
        grad_mu[:, kk] *= scale_factor
    return grad_mu

def compute_pred_cov_deriv(ps_t, xs_t, pred_cov, length_scales):
    npred_points = ps_t.shape[0]
    nbuild_points = xs_t.shape[0]
    ndims = xs_t.shape[1]
    pred_scaled_dists = np.zeros((npred_points, nbuild_points, ndims))
    for kk in range(ndims):
        # compute mixed (i.e. prediction - build) distances
        # and divide by length scale hyperparameters
        pred_scaled_dists[:, :, kk] = cdist(ps_t[:, kk].reshape(-1, 1),
            xs_t[:, kk].reshape(-1, 1),
            lambda u, v : u - v) / length_scales[kk]**2 # seems to return the same value as defualt metric...
    return np.einsum("ijk, ij -> ijk", -pred_scaled_dists, pred_cov)

make_plot = False
#fun = TestFunc.fun1D()
ndims = 1
xs, ys = get_nD_problem_data(ndims, np.array((0.1, 0.9)), 9, tf)
ps = np.atleast_2d(0.75) # test with multiple points


input_scaler = StandardScaler()
xs_t = input_scaler.fit_transform(xs)

output_scaler = StandardScaler()
ys_t = output_scaler.fit_transform(ys)

kernel_type = "SE"
skl_kernel = C(1.0, constant_value_bounds=(1e-2, 1e3)) \
    * partial_skl_kernel(ndims, kernel_type)
skl_gp = GaussianProcessRegressor(kernel=skl_kernel,
    n_restarts_optimizer=40, alpha=0.0, normalize_y=True, random_state=10)

skl_gp.fit(xs_t, ys)

ps_t = input_scaler.transform(ps)
mu = skl_gp.predict(ps_t, return_std=False)

# only need to compute alpha_vec once!
# eq. (6.14) in Dakota Theory manual: Gram @ alpha = z
# z (build data)
alpha_vec = cho_solve((skl_gp.L_, True), ys_t).squeeze()

# check on the computation of the GP mean
mu_scaled = np.atleast_2d(skl_gp.kernel_(ps_t, xs_t) @ alpha_vec).T  # eqs. (6.16) and (6.17) in Dakota Theory manual
mu_second_method = output_scaler.inverse_transform(mu_scaled) # scale back data to original representation

grad_mu = compute_gp_grad_mu(skl_gp, input_scaler, output_scaler, alpha_vec,
    xs_t, ps_t)

if make_plot:
    # plot fun, deriv, and data
    plt.close("all")
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(14, 12), sharex=True)
    fig.suptitle("Squared Exponential Kernel GP Gradient Check", fontsize=22, y=0.95)

    ax[0].plot(ps, tf(ps), color="blue", zorder=1, label="truth", linewidth=4)
    ax[0].scatter(xs, ys, color="black", marker="o", s=40, zorder=3, label="data")
    ax[0].set_xlabel("$x$", fontsize=18)
    ax[0].set_ylabel("$y(x)$", fontsize=18)
    #ax[0].plot(ps, mu, color="purple", zorder=2, label = "GP $\mu$")
    ax[0].plot(ps, mu_second_method, color="red", zorder=2, label = "GP $\mu$")
    ax[0].legend(loc="best", fontsize=14)

    eps = 1.0e-6
    gp_plus = skl_gp.predict(ps_t + eps, return_std=False)
    gp_minus = skl_gp.predict(ps_t - eps, return_std=False)

    gp_approx_deriv = (gp_plus - gp_minus) / (2. * eps * input_scaler.scale_[0])

    ax[1].plot(ps, d_tf(ps), color="blue", zorder=2, label="truth", linewidth=4)
    ax[1].plot(ps[1:-1, :], gp_approx_deriv[1:-1, :], color="gray", 
        linestyle="--", zorder=3, label="GP $\\nabla \mu$ (FD approx)")
    ax[1].plot(ps, grad_mu, color="red", zorder=2,
        label="GP $\\nabla \mu$ (computed)")
    ax[1].scatter(xs, d_tf(xs), color="black", marker="o", s=40, zorder=4,
        label="data")
    ax[1].set_xlabel("$x$", fontsize=18)
    ax[1].set_ylabel(r"$\frac{dy}{dx}$", fontsize=18)
    ax[1].legend(loc="best", fontsize=14)
    #plt.show()
