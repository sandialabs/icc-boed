import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from scipy.stats import qmc
from sklearn.metrics import mean_squared_error, mean_absolute_error
import utils as ut
import stats_utils as su

# DTS: this is not really specific to GPs. It could be used with any test function
def get_nD_data(l_bounds, u_bounds, nsamples, fun, seed=None,
    sampling_method='Halton', scramble=False, return_targets=True):

    """ given upper and lower parameter bounds and number of training
    points, generate training features and targets"""

    ndims = len(l_bounds)

    if sampling_method == 'Halton':
        sampler = qmc.Halton(d=ndims, scramble=scramble, seed=seed)
    elif sampling_method == 'Sobol':
        sampler = qmc.Sobol(d=ndims, scramble=scramble, seed=seed)
    elif sampling_method == 'LHC':
        sampler = qmc.LatinHypercube(d=ndims, seed=seed)
    else:
        raise NotImplementedError('Sampling method not implemented')

    samples = sampler.random(n=nsamples)
    train_features = qmc.scale(samples, l_bounds, u_bounds)
    if return_targets:
        train_targets = np.atleast_2d(fun(train_features))
        return train_features, train_targets
    else:
        return  train_features



def GP_build(features, targets, kernel_type, scale_input=True,
    scale_output=True, nrestarts=40, seed=40, alpha_val=1e-10,
    const_val_bnds=(1e-2, 1e3), len_scale_bnds=(1e-2, 1e2)):

    """ given features, targets, and other gp parameters,
    build GP
    """

    def partial_skl_kernel(ndim, kernel_type):
        init = (1.0,)
        if kernel_type == "SE":
            return RBF(init * ndim, length_scale_bounds=len_scale_bnds)
        else:
            raise NotImplementedError("gradient for kernel type not implemented")

    features = np.atleast_2d(features)
    ndims = features.shape[1]

    if scale_input:
        input_scaler = StandardScaler()
        features_t = input_scaler.fit_transform(features)
    else:
        input_scaler = None
        features_t = features

    if scale_output:
        output_scaler = StandardScaler()
        targets_t = output_scaler.fit_transform(targets)
    else:
        output_scaler = None

    gp_kernel = C(1.0, constant_value_bounds=const_val_bnds) * \
        partial_skl_kernel(ndims, kernel_type)

    gp = GaussianProcessRegressor(
        kernel=gp_kernel,
        n_restarts_optimizer=nrestarts,
        random_state=seed,
        alpha=alpha_val,
        normalize_y=scale_output)

    gp.fit(features_t, targets)

    return gp, input_scaler, output_scaler



def make_GP(npoints, l_bounds, u_bounds, fun,
    sampling_method='Halton', seed=10, scramble=False,
    scale_input=True, scale_output=True, nrestarts=40,
    alpha_val=1e-10):

    """ takes parameter bounds, function, and number of
    training points as input. Produces training data
    by calling 'get_nD_data'. Creates GP from training
    data by calling 'GP_build'
    """

    GP = {}
    # training data - targets and features
    GP['xs'], GP['ys'] = get_nD_data(\
        l_bounds, u_bounds, npoints, fun, seed=seed,
        sampling_method=sampling_method, scramble=scramble)

    # generate data on a finer grid for plotting
    GP['xgrid'], GP['ygrid'] = get_nD_data(\
        l_bounds, u_bounds, npoints*10, fun, seed=seed,
        sampling_method=sampling_method, scramble=scramble)

    GP['GP'], GP['input_scaler'], GP['output_scaler'] = \
        GP_build(GP['xs'], GP['ys'], "SE", scale_input=scale_input,
        scale_output=scale_output, nrestarts=nrestarts, seed=seed,
        alpha_val=alpha_val)

    return GP



def eval_GP(features, GP, input_scaler, return_std=False, scale=True):
    # separate input_scaler and features from GP
    # these are the same for all surrogates in tree for boed algorithm,
    #  so they are only saved once for all GPs


    if scale:
        features_t = input_scaler.transform(features)
    else:
        features_t = features

    gp_prediction = GP.predict(features_t, return_std=return_std)

    return gp_prediction.squeeze()



def GP_score(test_qoi, GP_qoi):
    u = ((test_qoi - GP_qoi)**2).sum()
    v = ((test_qoi - test_qoi.mean())**2).sum()
    score = 1 - (u/v)

    return score



def GP_quality(gp, input_scaler, test_features, train_features,
    test_targets, train_targets):
    """ move to using metrics in error_metrics.py instead"""

    gp_test_output = eval_GP(test_features, gp, input_scaler)
    test_score = gp.score(input_scaler.transform(test_features),
       test_targets.squeeze())
    gp_test_mse = mean_squared_error(test_targets.squeeze(),
       gp_test_output, multioutput='raw_values')
    gp_test_mae = mean_absolute_error(test_targets.squeeze(),
       gp_test_output, multioutput='raw_values')

    print("**********************")
    print("**   Test Quality   **")
    print("**********************")
    print(f"Score: {test_score}")
    print(f"MSE: {gp_test_mae}")
    print(f"MAE: {gp_test_mse}")



def get_GP_funs(GP, input_scaler, features, return_std=False):

    # analytical gradient of GP
    GP_analytical_grad = partial(\
        ut.analytical_GP_gradient,
        GP_dict=GP, input_scaler=input_scaler,
        build_pts=features)

    # functional form of GP with partial
    GP_func = partial(eval_GP,
                      GP=GP['GP'],
                      input_scaler=input_scaler,
                      return_std=return_std)

    return GP_analytical_grad, GP_func



def GP_grad_fun(data, noise_var, priors,
    fun_list, sign='neg', likelihood='normal',
    fd_eps=1e-6, grad_list=None, fmodel=None):

    grad_fun = partial(\
        su.compute_logpost_gradient,
        data=data,
        noise_var=noise_var,
        priors=theta_priors,
        fun_list=fun_list,
        sign='neg',
        fmodel=fmodel,
        likelihood=likelihood,
        grad=grad_list,
        fd_eps=fd_eps)


    return grad_fun
