import pickle
import numpy as np
import utility_functions
from posterior_update_tools import MCMC, Laplace_approx
from functools import partial
import qoi_models
import time
import copy
import sys
from utils import is_picklable
#from MTS_driver import reset_MTS_state

def run_boed(fmodel, experiment, nobs_pstep, theta_info,
    nsteps=5,  num_qois=None,
    design_options=None, design_initiation=None,
    utility_function='EIG', likelihood='normal',
    posterior_method='Laplace',  post_update_settings=None,
    sequential_design=True, noise_var=None, fd_eps=1e-4, fprime=None,
    continuous_design=True, nested_mc_settings=None,
    update_posterior=True, figdir=None, nobjectives=1, obs_inds=None,
    nInc=None, return_pca=False, verbose=False, use_objectives=None,
    surrogate_type='gps', surrogate_dir=None, exemplar='mps',
    standardize=False, integrate_displacement=False):
    """
    A continuous design is one such that the outcome
    of the experiment at step n is dependent on the outcome
    of experiments at previous steps. This is opposed to a discrete
    design (such as in the hot pepper example), where the outcome
    of each experiment is independent of previous experiments.

    Sequential design: the posterior becomes the prior (or sampling
    distribution) in EIG estimate
    """

    assert use_objectives is not None

    # confirm input strings
    if posterior_method != 'Laplace' and posterior_method != 'MCMC':
        raise ValueError('posterior method not implemented')

    if likelihood != 'normal':
        raise ValueError('likelihood function no implemented')

    if utility_function != 'EIG':
        raise ValueError('utility function not implemented')

    if nested_mc_settings is None:
        raise ValueError('must provide nested MC settings')

    mean_store = [None]*nsteps
    cov_store = [None]*nsteps
    post_store = [None]*nsteps
    EIG_store = [None]*nsteps
    map_store = [None]*nsteps

    pdfs = {}
    marginal_post_store = {}

    # dimensionality of parameter space
    D = len(theta_info['names'])
    ndes = len(design_options)
    # initiate array holding calibration data
    y_current = [None] * len(use_objectives)
    if return_pca or integrate_displacement:
        field_data = [None] * nsteps

    for idx, ob in enumerate(use_objectives):
        y_current[idx] = np.zeros((nsteps, nobs_pstep[ob], num_qois[ob]))

    # extract MCMC settings
    if posterior_method == 'MCMC':
        nsamples = int(post_update_settings['MCMC']['nsamples'])
        burnin = int(post_update_settings['MCMC']['burnin'])
        cov_check = post_update_settings['MCMC']['cov_check']
        target_accept = post_update_settings['MCMC']['target_accept']
        sampler = post_update_settings['MCMC']['sampler']
        prop_var = post_update_settings['MCMC']['prop_var']
        diagnostic_check = post_update_settings['MCMC']['diagnostic_check']
        iter_start_diag = post_update_settings['MCMC']['iter_start_diag']
        use_own_fd = post_update_settings['Laplace']['use_own_fd']

        trace = {"theta": np.zeros((nsamples, D, nsteps)),
                 "log_post": np.zeros((nsamples, nsteps))}

    if posterior_method == 'Laplace':
        fd_eps = post_update_settings['Laplace']['fd_eps']
        fprime = post_update_settings['Laplace']['fprime']
        use_own_fd = post_update_settings['Laplace']['use_own_fd']
        factr = post_update_settings['Laplace']['factr']
        nrestarts = post_update_settings['Laplace']['nrestarts']
        psd_check_methods = post_update_settings['Laplace']['psd_check_methods']
        fd_order = post_update_settings['Laplace']['fd_order']
        pgtol = post_update_settings['Laplace']['pgtol']
        maxls = post_update_settings['Laplace']['maxls']

        Laplace_summary = {}
        Laplace_summary['Hessian Adjusted'] = [None]*nsteps
        Laplace_summary['Hessian Nugget'] = [None]*nsteps
        Laplace_summary['Cov Adjusted'] = [None]*nsteps
        Laplace_summary['Cov Nugget'] = [None]*nsteps
        Laplace_summary['Hessian MAPE'] = [None]*nsteps
        Laplace_summary['Hessian MaxPE'] = [None]*nsteps
        Laplace_summary['Cov MAPE'] = [None]*nsteps
        Laplace_summary['Singularity Overridden'] = [None]*nsteps
        Laplace_summary['gradient at theta map'] = [None]*nsteps

    # initiate list that will hold the design
    # decision for each step
    current_design = [None]*nsteps

    # no posterior at initiation
    post = None

    # extract parameter information (priors, bounds, initiation)
    priors = theta_info['prior']

    theta_init = theta_info['init']

    if theta_info['bounds'] is not None:
        bounds = np.atleast_2d(np.asarray(theta_info['bounds']))
    else:
        bounds = None

    if theta_info['true'] is not None:
        theta_true = np.atleast_2d(theta_info['true'])
    else:
        theta_true = None


    if np.isnan(bounds).all():
        bounds = None

    theta_names = theta_info['names']

    # number of experiments pre-determined
    nexp_pre = len(design_initiation)
    assert nexp_pre <= nsteps

    # experiments that have been pre-determined
    exp_pre = [None]*nsteps
    exp_pre[:nexp_pre] = design_initiation
    priors_init = priors

    # prior distribution in EIG calculation
    EIG_prior_dist = copy.deepcopy(priors)

    for boe_step in range(nsteps):
        if verbose:
            print("*************************")
            print(f"BOED STEP: {boe_step}")
        step_start = time.time()
        # initiate design, if the initial experiment of the design is given,
        # then run the experiment and get data, if it is not given, then
        # calculate EIG to pick the initial experiment of the design

        if exp_pre[boe_step] is not None:
            # run pre-determined experiment
            current_design[boe_step] = exp_pre[boe_step]

            if continuous_design:
                exp_design = current_design[:boe_step+1]
            else:
                exp_design = [current_design[boe_step]]

            new_data = experiment(exp_design)
            if nobjectives == 1:
                y_current[0][boe_step, ...] = new_data[0][-nobs_pstep[0]:, :]
            else:
                for idx, ob in enumerate(use_objectives):
                    exp_data = new_data[idx]#[boe_step, ...]
                    y_current[idx][boe_step, ...] = exp_data
                if return_pca or integrate_displacement and 0 in use_objectives:
                    field_data[boe_step] = new_data[-1]
            design_EIGs = 'None'

        else:
            if verbose:
                print(f"Calculating EIG step: {boe_step}")

            # calculate EIG for each design option
            EIG_start = time.time()

#            try:
            design_EIGs =\
                utility_functions.EIG(current_design[:boe_step],
                design_options, D, boe_step,
                EIG_prior_dist, priors, fmodel,
                fd_eps=fd_eps, use_own_fd=use_own_fd,
                fprime=fprime, bounds=bounds,
                posterior_method=posterior_method,
                sequential_design=sequential_design,
                likelihood=likelihood,
                noise_var=noise_var,
                continuous_design=continuous_design, num_qois=num_qois,
                nested_mc_settings=nested_mc_settings,
                nobjectives=nobjectives, use_objectives=use_objectives,
                return_pca=return_pca,
                theta_0=theta_init,
                surrogate_type=surrogate_type,
                surrogate_dir=surrogate_dir,
                standardize=standardize,
                exemplar=exemplar,
                integrate_displacement=integrate_displacement)
            EIG_end = time.time()
#            except:
#                print("Problem computing EIG --- saving workspace to 'backup.pkl' and exiting python")
#                pkl_name = 'backup.pkl'
#                bk = {}
#                for k in dir():
#                    obj = locals()[k]
#                    try:
#                        bk.update({k: obj})
#                    except TypeError:
#                        pass
#
#                with open(pkl_name, 'wb') as f:
#                    pickle.dump(bk, f)
                #reset_MTS_state()
                #sys.exit()
            if verbose:
                print(f"time to calculate EIG: {(EIG_end - EIG_start)/60} min")

            optimal_idx = np.argmax(design_EIGs)
            optimal_design = design_options[optimal_idx]
            current_design[boe_step] = optimal_design
            if verbose:
                print(f"EIG design A: {design_EIGs[0]},   EIG design B: {design_EIGs[1]}")
                print(f"current design: {current_design}")
            if continuous_design:
                exp_design = current_design[:boe_step+1]
            else:
                exp_design = [current_design[boe_step]]

            # run experiment
            new_data = experiment(exp_design)
            if nobjectives == 1:
                y_current[0][boe_step, ...] = new_data[0][-nobs_pstep[0]:, :]
            else:
                for idx, ob in enumerate(use_objectives):
                    y_current[idx][boe_step, ...] = new_data[idx]#[boe_step, ...]
                if return_pca or integrate_displacement and 0 in use_objectives:
                    field_data[boe_step] = new_data[-1]

        # where the calibration happens
        if update_posterior:
            # posterior update
            # create parital function with the design
            # fixed to the current state of the design

            data = [None] * len(use_objectives)
            if exemplar == 'mps': #mps case
                fmodel_des_fixed = partial(fmodel, design=exp_design, return_steps='all')
                data[0] = np.atleast_2d(y_current[0][:boe_step+1, ...])
            elif exemplar == 'cruciform': #cruciform case
                surr = qoi_models.load_cruciform_surrogate(\
                    exp_design, surrogate_type, return_steps='all',
                    surrogate_dir=surrogate_dir)
                fmodel_des_fixed = partial(fmodel, surr=surr)
                for idx, ob in enumerate(use_objectives):
                    data[idx] = np.atleast_2d(y_current[idx][:boe_step+1, ...])

#            if sequential_design:
#                # fmodel returns output at last design decision
#                fmodel_des_fixed = partial(fmodel, design=exp_design, return_steps='last')
#                data = np.atleast_2d(y_current[boe_step,:])
#            else:
#                # fmodel returns output from all design decisions
#                fmodel_des_fixed = parital(fmodel, design=exp_design, return_steps='all')
#                data = np.atleast_2d(y_current[:boe_step+1,:])

            if False:
                # plot data at each step
                print(data)
                plot_obs = (boe_step+1)*nInc - (boe_step+1)
                plot_obs_inds = np.arange(plot_obs)
                data_obs_inds = obs_inds[obs_inds <= plot_obs]
                plot_model = partial(fmodel, obs_inds=plot_obs_inds, design=exp_design, return_steps='all')
                y_plot = plot_model(theta_true)
                import matplotlib.pyplot as plt
                import matplotlib
                plt.figure()
                plt.plot(plot_obs_inds, y_plot[0, :, 0])
                plt.plot(plot_obs_inds, y_plot[0, :, 1])
                plt.plot(data_obs_inds, data[:,:,0].reshape(-1, 1), '.', ms=20)
                plt.plot(data_obs_inds, data[:,:,1].reshape(-1, 1), '.', ms=20)
                plt.vlines(x=data_obs_inds[nobs_pstep-1::nobs_pstep],
                    ymin=0, ymax=500, color='black', linestyle='dashed')
                plt.show()

            if posterior_method == 'Laplace':

                if verbose:
                    print("Calculating Posterior")

                try:
                    post, theta_map, cov, log_evidence, Laplace_summary =\
                        Laplace_approx(data, fmodel_des_fixed,
                        noise_var, priors, theta_0=theta_init,
                        bounds=bounds,
                        fprime=fprime, fd_eps=fd_eps,
                        likelihood=likelihood,
                        use_own_fd=use_own_fd,
                        factr=factr, boe_step=boe_step,
                        Laplace_summary=Laplace_summary,
                        nrestarts=nrestarts,
                        psd_check_methods=psd_check_methods,
                        fd_order=fd_order,
                        pgtol=pgtol,
                        maxls=maxls,
                        nobjectives=nobjectives,
                        use_objectives=use_objectives,
                        return_pca=return_pca,
                        design=exp_design,
                        num_qois=num_qois,
                        return_steps='all',
                        exemplar=exemplar,
                        standardize=standardize,
                        integrate_displacement=integrate_displacement)

                except TypeError:
                    print("loading pre-calculated posterior")
                    with open('../data/boed_results/step_1_posterior.pkl', 'rb') as f:
                        post_1 = pickle.load(f)
                    post = copy.deepcopy(post_1)
                    theta_map = post_1.mean.copy()
                    cov = post_1.cov.copy()


                post_mean = theta_map.copy()
                post_cov = cov.copy()
                theta_init = theta_map.copy()
                if verbose:
                    print(f"Calculated Posterior Mean: {post_mean}")
                    print(f"Calculated Posterior Marginal Var: {np.diag(cov)}")

            elif posterior_method == 'MCMC':
                Laplace_summary = None
                trace, prop_var = MCMC(data, priors, fmodel_des_fixed,
                    nsamples, burnin, cov_check, sampler, prop_var,
                    bounds, theta_init, noise_var, likelihood,
                    diagnostic_check, iter_start_diag, theta_true,
                    figdir, boe_step, nobjectives=nobjectives,
                    use_objectives=use_objectives,
                    return_pca=return_pca, design=exp_design,
                    num_qois=num_qois, integrate_displacement=integrate_displacement,
                    exemplar=exemplar)

                post_cov = np.cov(trace['theta'][burnin:,:].T)
                post_mean = np.mean(trace['theta'][burnin:,:], axis=0)
                post = trace['theta'][burnin:,:]
                map_index = np.argmax(trace['log_post'])
                theta_map = trace['theta'][map_index]

                theta_init = theta_map

        else: # if posterior not updated
            post = None
            theta_map = None
            post_mean = None
            post_cov = None

        if sequential_design:
            EIG_prior_dist = [post]

        post_store[boe_step] = post
        map_store[boe_step] = theta_map
        mean_store[boe_step] = post_mean
        cov_store[boe_step] = post_cov
        EIG_store[boe_step] = design_EIGs
        step_end = time.time()
        if verbose:
            print(f"total step time: {(step_end - step_start)/60} min")

    if not return_pca and not integrate_displacement and 0 in use_objectives:
        field_data = y_current[0]
    elif 0 not in use_objectives:
        field_data = None

    # store settings in dictionary
    boed_dict = {'Nested MC Settings': nested_mc_settings, 'par names': theta_names,
        'true par values': theta_true,
        'nsteps': nsteps, 'design initiation': design_initiation,
        'FD eps': fd_eps,
        'Posterior Method': posterior_method,
        'Sequential Design': sequential_design,
        'Continuous Design': continuous_design,
        'Update Posterior': update_posterior,
        'Noise variance': noise_var,
        'Prior Distribution': priors_init,
        'Data': y_current, 'Design': current_design,
        'Post': post_store, 'Post Mean': mean_store,
        'Cov': cov_store, 'EIG': EIG_store,
        'Laplace Summary': Laplace_summary,
        'Field Data': field_data}

    return boed_dict

