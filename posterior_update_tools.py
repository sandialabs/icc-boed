import numpy as np
from stats_utils import unnorm_log_post, importance_samples
from scipy.stats import norm, uniform, multivariate_normal
from functools import partial
from scipy.optimize import fmin_l_bfgs_b
from utils import compute_fd_hessian, compute_fd_gradient, compute_centered_fd_epsilon
from MCMC_diagnostic_figures import diagnostics
from sklearn.preprocessing import StandardScaler
from utils import isPD, nearestPD, get_near_psd
from numpy import linalg as la
import copy
import qoi_models
from full_cov_mat_adj import cov_mat_adj
import time

def extract_posterior_method_settings(posterior_method, post_update_settings):

    if posterior_method == 'MCMC':
        nsamples = int(post_update_settings['MCMC']['nsamples'])
        burnin = int(post_update_settings['MCMC']['burnin'])
        cov_check = post_update_settings['MCMC']['cov_check']
        target_accept = post_update_settings['MCMC']['target_accept']
        sampler = post_update_settings['MCMC']['sampler']
        prop_var = post_update_settings['MCMC']['prop_var']
        diagnostic_check = post_update_settings['MCMC']['diagnostic_check']
        iter_start_diag = post_update_settings['MCMC']['iter_start_diag']

        trace = {"theta": np.zeros((nsamples, D, nsteps)),
                 "log_post": np.zeros((nsamples, nsteps))}

        return nsamples, burnin, cov_check, target_accept,\
            sampler, prop_var, diagnostic_check, iter_start_diag


    elif posterior_method == 'Laplace':
        fd_eps = post_update_settings['Laplace']['fd_eps']
        fprime = post_update_settings['Laplace']['fprime']
        use_own_fd = post_update_settings['Laplace']['use_own_fd']
        factr = post_update_settings['Laplace']['factr']
        nrestarts = post_update_settings['Laplace']['nrestarts']
        psd_check_methods = post_update_settings['Laplace']['psd_check_methods']
        fd_order = post_update_settings['Laplace']['fd_order']
        pgtol = post_update_settings['Laplace']['pgtol']
        lnsearch = post_update_settings['Lapalce']['lnsearch']

        return fd_eps, fprime, use_own_fd, factr, nrestarts,\
            psd_check_methods, fd_order, pgtol, lnsearch



def MCMC(data, theta_prior, fmodel, nsamples=10000, burnin=2000, cov_check=1000,
    sampler='metropolis_hastings', prop_var=.01, bounds=None,
    theta_0=None, noise_var=None, likelihood='normal', diagnostic_check=10000,
    iter_start_diag=2000, theta_true=None, figdir=None, boe_step=None,
    nobjectives=1, use_objectives=None, return_pca=False, design=None,
    num_qois=None, exemplar='mps', integrate_displacement=False
    ):
    """
    Input:
        theta_priors -- type pya.variables.joint.IndependentMarginalsVariable()
        and contains marginal priors for unknown forward model parameters
        sampler -- string, MCMC sampler
        data -- SxN array of observations
        fmodel -- forward model that maps parameters to quantities of interest
        theta_0 -- (optional) initiation point, if not provided, the chaing will
            initiate with a random sample from the prior
        nsamples -- scaler, run algorithm until chain is of length len_chain
        noise_var (observation noise variance) -- if None, then include in the inference problem
                     if scaler, i.i.d observation error assumed
                     if (1xN) array, different obervation error for each observations
        burnin -- scaler, number of chain samples to treat as burnin (eventually include internal
            diagnostics for chain convergence)
        cov_check -- scaler, how often to check covariance of chain and adapt proposal variance (improves mixing)
        bnds -- list of lenght D with parameter bounds for sampling (optional)
                'None' should be used for parameters with no bounds


    Output:
        MCMC chain targeting the posterior distribution
        of length 'len_chain'
    """

    # eventually include other samplers, and Gibbs sampling for
    # parameters with conjugate priors/full-conditional distributions
    if sampler != 'metropolis_hastings':
       raise ValueError("sampler not yet implemented")

    if noise_var is None:
       raise valueError("noise variance not yet incorporated into inference problem.")
#    noise_var = np.atleast_1d(np.asarray(noise_var))

    if return_pca and 0 in use_objectives:
        disp_noise_var = qoi_models.pca_noise_var(design,
            noise_var[0], num_qois[0], return_steps='all')
        noise_var = [disp_noise_var, noise_var[1]]

    if nobjectives == 1:
        noise_var[0] = np.atleast_1d(noise_var[0])
        if np.ndim(noise_var[0]) == 1:
            data[0] = np.atleast_2d(data[0])
            nqoi = data[0].shape[-1]
            nload_steps = data[0].shape[0]
            err_pr = [np.full((nload_steps, nqoi), 1/noise_var[0])]
        elif np.ndim(noise_var[0]) > 1:
            err_pr = [np.linalg.inv(noise_var[0])]
    else:
        err_pr = [None] * len(use_objectives)
        for idx, ob in enumerate(use_objectives):
            data[idx] = np.atleast_2d(data[idx])
            nqoi = data[idx].shape[-1]
            nload_steps = data[idx].shape[0]
            nv = np.atleast_1d(noise_var[ob])
            if np.ndim(nv) == 1:
                err_pr[idx] = np.full((nload_steps, nqoi), 1/nv)
            elif np.ndim(nv) > 1:
                err_pr[idx] = np.linalg.inv(nv)

    target_accept = np.array([0.2, 0.5])
    # chain initiation point
    if theta_0 is None:
        if len(theta_prior) == 1:
            D = theta_prior[0].rvs().shape[0]
            theta_0 = importance_samples(\
                theta_prior[0], 1, D, bounds)
        else:
            D = len(theta_prior)
            theta_0 = np.zeros((D))
            for p, prior in enumerate(theta_prior):
                theta_0[p] = np.atleast_2d(\
                    importance_samples(\
                    prior, 1, 1, bounds=bounds[p]))

    theta_0 = np.atleast_2d(theta_0)
    if theta_true is not None:
        theta_true = np.atleast_2d(theta_true)
    D = theta_0.shape[1]
    curr_theta = theta_0.copy()

#    prop_var = np.atleast_1d(np.asarray(prop_var))
    prop_cov_mat = np.eye(D) * prop_var
    # if the proposal variance is not defined for each parameter,
    # the first value is used for all parameters
#    if len(prop_var) != D:
#        prop_var = np.repeat(prop_var[0], D)

    # bounds for sampling
    #if bounds is None:
    #    bounds = [None] * D
    #elif len(bounds) is not D:
    #    raise ValueError("Parameter bounds must be defined for all parameters")
    lbs = [bounds[dd][0] for dd in np.arange(D)]
    ubs = [bounds[dd][1] for dd in np.arange(D)]

    theta_chain = np.zeros((nsamples,D))
    log_post = np.zeros(nsamples)

    # track paramter acceptance rate for each parameter
#    acceptance = np.zeros((nsamples,D))
    acceptance = np.zeros(nsamples)

    # initial log posterior calculation
    curr_post = unnorm_log_post(
        curr_theta, data, err_pr,
        theta_prior, fmodel=fmodel, likelihood=likelihood,
        nobjectives=nobjectives, use_objectives=use_objectives,
        return_pca=return_pca, exemplar=exemplar)

    start_time = time.time()
    for sample in range(nsamples):

        if np.mod(sample, 500) == 0 :
            print(f"sample = {sample}")

        # covariance adaptation
        if np.mod(sample, cov_check) == 0 and sample > 0 and sample <= burnin:
            accept_rate = (1 + acceptance[sample - cov_check : sample-1].sum(axis=0)) / cov_check
            print(f"Accept rate: {accept_rate}")
#            for dd in range(D):
#                if accept_rate[dd] < target_accept[0] or accept_rate[dd] > target_accept[1]:
#                    prop_var[dd] = prop_var[dd] * accept_rate[dd] / \
#                       (target_accept[0] + np.ptp(target_accept[1]) / 2)
            if accept_rate < target_accept[0] or accept_rate > target_accept[1]:
                temp_chain = theta_chain[sample-cov_check:sample, ...]
                C = np.cov(temp_chain.T)
                cov_diag = np.diag(C)*accept_rate / \
                    (target_accept[0] + np.ptp(target_accept) /2)
                np.fill_diagonal(prop_cov_mat, cov_diag)
                prop_cov_mat = cov_mat_adj(temp_chain, prop_cov_mat)

        prop_theta = curr_theta.copy()
#        for dd in range(D):

#            while True:
                # sample a new theta from proposal distribution
#                prop_theta[0,dd] = norm(curr_theta[0,dd], prop_var[dd] ** 0.5).rvs(1)
#                if bounds[dd] is None:
#                    break
#                elif bounds[dd][0] <= prop_theta[0,dd] <= \
#                    bounds[dd][1]:
#                    break

        while True:
            # sample a new theta vector from proposal distribution
            prop_theta = multivariate_normal(curr_theta.squeeze(), prop_cov_mat).rvs(1)
            if bounds is None:
                break
            elif np.all(lbs <= prop_theta) & np.all(prop_theta <= ubs):
                break
        prop_theta = np.atleast_2d(prop_theta)
        prop_post = unnorm_log_post(prop_theta,\
            data, err_pr, theta_prior,
            fmodel=fmodel, likelihood=likelihood,
            nobjectives=nobjectives,
            use_objectives=use_objectives,
            return_pca=return_pca, exemplar=exemplar)

        if np.log(uniform(0,1).rvs(1)) < prop_post - curr_post:
#            curr_theta[0,dd] = prop_theta[0,dd].copy()
            curr_theta = prop_theta.copy()
            curr_post = prop_post.copy()
            acceptance[sample] = 1
#            acceptance[sample][dd] = 1
        else:
             prop_theta = curr_theta.copy()
             acceptance[sample] = 0
 #           prop_theta[0,dd] = curr_theta[0,dd].copy()
 #           acceptance[sample][dd] = 0

        theta_chain[sample] = curr_theta
        log_post[sample] = curr_post

        if np.mod(sample, diagnostic_check) == 0 and sample > iter_start_diag:
            trace = {}
            trace['theta'] = theta_chain
            trace['log_post'] = log_post
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time for last {diagnostic_check} iterations: {elapsed_time/60} minutes.")
            diagnostics(trace, sample, figdir, boe_step,
                theta_true, iter_start_diag)
            start_time = time.time()

    trace = {}
    trace['theta'] = theta_chain
    trace['log_post'] = log_post
    return trace, prop_var


def Laplace_approx(data, fmodel, noise_var,
    priors, theta_0=None, bounds=None, fprime=None, fd_eps=1e-6,
    likelihood='normal', use_own_fd=True, factr=1e7,
    boe_step=None, Laplace_summary=None, nrestarts=[3,3],
    psd_check_methods='all', fd_order=2, pgtol=1e-5,
    maxls=20, nobjectives=1, use_objectives=None,
    return_pca=False, design=None,
    num_qois=1, return_steps='all',
    override_singular=True,
    exemplar='mps', standardize=False,
    integrate_displacement=False):

    if Laplace_summary is None:
        track_adjustments = False
    else:
        track_adjustments = True

    # set approx_grad for bfgs based on whether fprime is provided
    approx_grad = False
    if fprime is None:
        approx_grad = True

    if return_pca and 0 in use_objectives:
        disp_noise_var = qoi_models.pca_noise_var(design,
            noise_var[0], num_qois[0], return_steps=return_steps)
        noise_var = [disp_noise_var, noise_var[1]]

    if exemplar == 'mps':
        noise_var[0] = np.atleast_1d(noise_var[0])
        if np.ndim(noise_var[0]) == 1:
            data[0] = np.atleast_2d(data[0])
            nqoi = data[0].shape[-1]
            nload_steps = data[0].shape[0]
            err_pr = [np.full((nload_steps, nqoi), 1/noise_var[0])]
        elif np.ndim(noise_var[0]) > 1:
            err_pr = [np.linalg.inv(noise_var[0])]
    elif exemplar == 'cruciform':
        err_pr = [None] * len(use_objectives)
        for idx, ob in enumerate(use_objectives):
            data[idx] = np.atleast_2d(data[idx])
            nqoi = data[idx].shape[-1]
            nload_steps = data[idx].shape[0]
            nv = np.atleast_1d(noise_var[ob])
            if np.ndim(nv) == 1:
                err_pr[idx] = np.full((nload_steps, nqoi), 1/nv)
            elif np.ndim(nv) > 1:
                err_pr[idx] = np.linalg.inv(nv)

    obj_func = partial(unnorm_log_post,
                       data=data,
                       err_pr=err_pr,
                       priors=priors,
                       sign='neg',
                       fmodel=fmodel,
                       likelihood=likelihood,
                       nobjectives=nobjectives,
                       use_objectives=use_objectives,
                       return_pca=return_pca,
                       exemplar=exemplar,
                       standardize=standardize)



    # if using our own implementation of finite difference for
    # gradient calculations in bfgs
    if use_own_fd:
        fprime = partial(\
            compute_fd_gradient,
            func=obj_func,
            fd_eps=fd_eps,
            order=fd_order)
        approx_grad = False

    if theta_0 is None:
        init_theta_0 = False
    else:
        init_theta_0 = True
        theta_0_bk = theta_0.copy()

    # do minimization for nrestarts times and save the mininum state
    nrestarts_min = nrestarts[0]
    nrestarts_max = nrestarts[1]
    for start in np.arange(nrestarts_max):
        if not init_theta_0 or start > 0:
            if len(priors) == 1:
                D = np.atleast_1d(priors[0].rvs()).shape[0]
                theta_0 = importance_samples(\
                    priors[0], 1, D, bounds)
            else:
                D = len(priors)
                theta_0 = np.zeros(D)
                for p, prior in enumerate(priors):
                    theta_0[p] = importance_samples(\
                        prior, 1, 1, bounds=bounds[p])

        theta_0 = np.atleast_2d(theta_0)
        D = theta_0.shape[1]

        if fd_eps == 'auto':
            if bounds is not None:
                fdeps = compute_centered_fd_epsilon(\
                    np.atleast_2d(bounds[:, 1]),
                    order=fd_order).squeeze()
                fdeps_hess = compute_centered_fd_epsilon(\
                    np.atleast_2d(bounds[:, 1]),
                    order=fd_order, approx_type='hessian')
            else:
                fdeps = compute_centered_fd_epsilon(theta_0,
                    order=fd_order).squeeze()
                fdeps_hess = compute_centered_fd_epsilon(theta_0,
                    order=fd_order, approx_type='hessian')
        else:
            fdeps = np.diag(np.eye(D) * fd_eps)
            fdeps_hess = np.atleast_2d(fdeps*1e2)

        # bring parameter bounds in slightly to avoid error in grad
        # and hessian calculation, which require model evaluations
        # to the left and right of the central point

        # lists are mutable objects, make deep copy before altering
        # otherwise it will alter outside the function as well
        par_bounds = copy.deepcopy(bounds)
        if True:
            if par_bounds is not None:
                for d in np.arange(D):
                    if par_bounds[d] is not None and ~np.isnan(par_bounds[d]).any():
                        # adjust parameter bounds for finite differencing in Laplace approx
                        par_bounds[d, 0] = par_bounds[d, 0] + 2*fdeps_hess[0, d]
                        par_bounds[d, 1] = par_bounds[d, 1] - 2*fdeps_hess[0, d]

        # optimization of objective function
        theta_map_temp, fun, cvg_dict = fmin_l_bfgs_b(obj_func,
            theta_0, bounds=par_bounds, fprime=fprime,
            approx_grad=approx_grad, epsilon=fd_eps,
            factr=factr, pgtol=pgtol, maxls=maxls)

#        print("***********************")
#        print("***Laplace Info *******")
#        print("***********************")

#        print(f"Laplace start {start}")
#        print(f"initiation location: {theta_0}")
#        print(f"theta_map: {theta_map_temp}")
#        print(f"gradient at theta map: {cvg_dict['grad']}")
#        print(f"log posterior at theta map: {obj_func(theta_map_temp)}")
        obj_func(theta_map_temp)

        hess_temp = compute_fd_hessian(\
            theta_map_temp,
            obj_func,
            fd_eps=fd_eps,
            order=fd_order)
        hess_psd_temp = isPD(hess_temp)
#        print(f"hessian PSD: {hess_psd_temp}")
        fun_min_tol = 1
        if start == 0:
            min_val = fun
            theta_map = theta_map_temp
            cvg = cvg_dict.copy()
            hess_psd = hess_psd_temp
        elif fun < min_val and hess_psd - hess_psd_temp != 1:
            theta_map = theta_map_temp
            min_val = fun
            cvg = cvg_dict.copy()
            hess_psd = hess_psd_temp
        elif np.abs(fun - min_val) < fun_min_tol and hess_psd - hess_psd_temp == -1:
            theta_map = theta_map_temp
            min_val = fun
            cvg = cvg_dict.copy()
            hess_psd = hess_psd_temp

        if start >= nrestarts_min-1 and hess_psd:
            break


    #print(f"theta_map: {theta_map}")
    #print(f"gradient at theta map: {cvg['grad']}")
    #print(f"PSD hessian: {hess_psd}")
    #print(f"log posterior at theta map: {obj_func(theta_map_temp)}")
    # compute hessian at MAP value
    hessian = compute_fd_hessian(\
        theta_map,
        obj_func,
        fd_eps=fd_eps,
        order=fd_order)

    hessian_bk = hessian.copy()
    # if hessian not positive definite, find nearest PD matrix
    if not isPD(hessian, methods=psd_check_methods):
        hessian_psd = nearestPD(hessian)
        hessian2_psd = get_near_psd(hessian)
        if not isPD(hessian_psd, methods=psd_check_methods):
            print("could not find near psd matrix for hessian")
        else:
            if track_adjustments:
                Laplace_summary['Hessian Adjusted'][boe_step] = True
            hessian = hessian_psd
    else:
        if track_adjustments:
            Laplace_summary['Hessian Adjusted'][boe_step] = False


    # add nugget to hessian to avoid singularity
    det_tol = 1e-10
    count_max = 1e5
    count = 0
    if la.det(hessian) < det_tol:
        while la.det(hessian) < det_tol:
            hessian += np.eye(D) * 1e-10
            count += 1
            if count > count_max:
                break
        if track_adjustments:
            Laplace_summary['Hessian Nugget'][boe_step] = count*1e-10
    hess_mape = np.divide(np.abs(hessian - hessian_bk), np.abs(hessian)).mean()*100
    hess_maxpe = np.divide(np.abs(hessian - hessian_bk), np.abs(hessian)).max()*100
    if track_adjustments:
        Laplace_summary['Hessian MAPE'][boe_step] = hess_mape
        Laplace_summary['Hessian MaxPE'][boe_step] = hess_maxpe
    det_hessian = la.det(hessian)

    cov = la.inv(hessian).squeeze()
    cov_bk = cov.copy()
    # it is possible for the hessian to be PSD but have an
    # inverse that is not
    if True:
        # if covariance is not positive semidefinite, get
        # near psd matrix
        if not isPD(cov, methods=psd_check_methods):
            cov_psd = nearestPD(cov)
            if not isPD(cov_psd, methods=psd_check_methods):
                print("could not find near psd for cov")
            else:
                if track_adjustments:
                    Laplace_summary['Cov Adjusted'][boe_step] = True
                cov = cov_psd
        else:
            if track_adjustments:
                Laplace_summary['Cov Adjusted'][boe_step] = False

    if False:
        count = 0
        if la.det(cov) <= det_tol:
            while la.det(cov) <= det_tol:
                cov += np.eye(D) * 1e-10
                count += 1
            if track_adjustments:
                Laplace_summary['Cov Nugget'][boe_step] = count*1e-10

    cov_mape = np.divide(np.abs(cov - cov_bk), cov).mean()*100
    if track_adjustments:
        Laplace_summary['Cov MAPE'][boe_step] = cov_mape
        Laplace_summary['gradient at theta map'][boe_step] = cvg['grad']

    try:
        posterior = multivariate_normal(theta_map, cov)
        if track_adjustments:
            Laplace_summary['Singularity Overridden'][boe_step] = False

        # evidence calculation
        log_f_theta_map = - obj_func(theta_map)

        log_evidence = log_f_theta_map + (D/2)* np.log(2*np.pi)\
            - 0.5 *  np.log(det_hessian)

    except:
        if override_singular:
            if isPD(cov, methods=psd_check_methods):
                # covariance is poorly conditioned but is PD
                if track_adjustments:
                    Laplace_summary['Singularity Overridden'][boe_step] = True
                posterior = multivariate_normal(\
                    theta_map, cov, allow_singular=True)

                # evidence calculation
                log_f_theta_map = - obj_func(theta_map)

                log_evidence = log_f_theta_map + (D/2)* np.log(2*np.pi)\
                    - 0.5 *  np.log(det_hessian)
            else:
                # covariance is not PD
                posterior = None
                log_evidence = None
        else:
            posterior = None
            log_evidence = None

    if track_adjustments:
        return posterior, theta_map, cov, log_evidence, Laplace_summary
    else:
        return posterior, theta_map, cov, log_evidence
