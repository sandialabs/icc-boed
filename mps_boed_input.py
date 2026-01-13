from scipy.stats import norm, uniform, multivariate_normal
import numpy as np

import random
import os
import time
import pickle

from evaluate_models import compute_qois

import seaborn as sns; sns.set()
from functools import partial
from tqdm import tqdm

# my modules
import utils as ut
import qoi_models as qm
import stats_utils as stu
import warnings
from boed_alg import run_boed
from posterior_update_tools import Laplace_approx, MCMC

import matplotlib.pyplot as plt
import matplotlib

from sklearn.preprocessing import StandardScaler
#warnings.simplefilter('error')
#plt.rcParams["text.usetex"] = False

# set random seed
np.random.seed(1)

# set up fig dir
workdir = os.getcwd()
dirname = f"{workdir}/boed_results/figures"
figdir_exists = os.path.isdir(dirname)
if not figdir_exists:
    os.makedirs(dirname)

build_data_dir = "surrogate_data/"

# BOED settings
nsteps = 5

#Nested MC Settings
M = 10 ** 4
N = 10 ** 4

# if N not large enough when using a self-normalizing biasing
# distribution, it could result in a q_evd with a covariance
# matrix that is extremely narrow (if the point is on the
# outskirts, resulting in 0 values for
# q_evd.pdf(samples)

# laplace_based and self_normalizing biasing distributions
# default to indepenent inner and outer samples, as well as
# independent inner samples (M) for each N

ind_inner_samples = False # independent M samples for each N
ind_inner_outer_samples = False # M and N samples independent of each other

evidence_method = 'importance_sampling' # options: 'Laplace', 'importance_sampling'
biasing_dist = 'prior' #options: None, 'prior', 'laplace_based', 'self-normalizing'
nested_mc_settings = {}
nested_mc_settings['M'] = M
nested_mc_settings['N'] = N
nested_mc_settings['ind_inner_samples'] = ind_inner_samples
nested_mc_settings['ind_inner_outer_samples'] = ind_inner_outer_samples
nested_mc_settings['evidence_method'] = evidence_method
nested_mc_settings['biasing_dist'] = biasing_dist

# define parameter names, true value (if available),
# bounds (if known), and priors
# if true value and/or bounds not known, define as 'None'

design_mode = 'adaptive'
design_initiation = ['d1']

theta_names = ["F", "G", "A", "n", "sigY"]
D = len(theta_names)

theta_true = np.array(([0.69, 0.43, 100. ,20., 300.]))

par_scaler = StandardScaler()
theta_true_scaled = par_scaler.fit_transform(np.atleast_2d(\
    theta_true).T)
par_scaler = None

theta_bounds = [[0.3, 0.7], [0.3, 0.7], [10, 400], [1e-6, 100.], [50., 500.]]
theta_0 = None #np.array(([0.5, 0.5, 200., 50., 250.]))

# 'underflow' problem mentioned in literature: when
# uniform prior is used, the likelihood can be -Inf
# at sampled thetas

prior_dist = ['trunc_normal'] * D
prior_mean = [0.5, 0.5, 200., 50., 250.]
#prior_mean = list(theta_true.squeeze())

# with diffuse prior, there are times when a PSD hessian
# inverts to a non-PSD covariance matrix
# I did not see this behavior with a less diffuse prior
# std = [1, 1, 200, 50, 200]
prior_std = [1, 1, 1e3, 1e2, 1e3]

if par_scaler is not None:
# look at some different scalar options
    bounds_array = np.asarray(theta_bounds)
    scaled_bounds_array = np.zeros_like(bounds_array)
    scaled_bounds_array[:,0] = \
        par_scaler.transform(np.atleast_2d(\
        bounds_array[:,0]).T).squeeze()
    scaled_bounds_array[:,1] = \
        par_scaler.transform(np.atleast_2d(\
        bounds_array[:,1]).T).squeeze()

    if theta_0 is not None:
        scaled_theta_0 = par_scaler.transform(np.atleast_2d(theta_0).T)
        theta_0 = scaled_theta_0.squeeze()

    theta_bounds = [list(row) for row in scaled_bounds_array]
    theta_true = theta_true_scaled.squeeze()

    prior_mean_array = np.asarray(prior_mean)
    scaled_prior_mean_array = par_scaler.transform(\
        np.atleast_2d(prior_mean_array).T).squeeze()
    prior_mean = list(scaled_prior_mean_array.squeeze())

#    prior_mean = list(theta_true.squeeze())
#    prior_mean = [0, 0, 0, 0, 0]
    prior_std = [1, 1, 1, 1, 1]


if prior_dist[0] != 'multivariate_normal':
    theta_priors = [None] * D
    for d in np.arange(D):
        if prior_dist[d] == 'trunc_normal':
            theta_priors[d] = \
            stu.get_truncated_normal(\
            mean=prior_mean[d],
            sd=prior_std[d],
            low=theta_bounds[d][0],
            upp=theta_bounds[d][1])

        elif prior_dist[d] == 'normal':
            theta_priors[d] = \
            norm(prior_mean[d], prior_std[d])

        elif prior_dist[d] == 'uniform':
            par_range = theta_bounds[d][1] - theta_bounds[d][0]
            theta_priors[d] = \
            uniform(theta_bounds[d][0],
                par_range)
else:

    pr_mean = np.asarray(prior_mean)
    pr_var = np.eye(D) * np.square(\
        np.asarray(prior_std))
    if not ut.isPD(pr_var):
        pr_var = nearest_PD(pr_var)

    det_tol = 1e-10
    if np.linalg.det(pr_var) < det_tol:
        while np.linalg.det(pr_var) < det_tol:
            pr_var += np.eye(D) * 1e-10

    try:
        theta_priors = [multivariate_normal(\
            pr_mean, pr_var)]
    except:
         if ut.isPD(pr_var):
             theta_priors = [\
                 multivariate_normal(\
                 pr_mean, pr_var, allow_singular=True)]
         else:
             raise ValueError('Prior covariance not PD')


if False:
    fix, ax = plt.subplots(1, D)
    for d in np.arange(D):
        min_val = theta_bounds[d][0]
        max_val = theta_bounds[d][1]
        xgrid = np.linspace(min_val, max_val, 1000)
        ax[d].plot(xgrid, theta_priors[d].pdf(xgrid))
        ax[d].plot(theta_true[d],
            theta_priors[d].pdf(xgrid).min(), 'r*', ms=15)
        plt.title(f"Log Prior for {theta_names[d]}")
    plt.show()

#for p, par in enumerate(theta_names):
theta_info = {}
theta_info['true'] = theta_true
theta_info['bounds'] = theta_bounds
theta_info['prior'] = theta_priors
theta_info['init'] = theta_0
theta_info['names'] = theta_names


# set up experiment:
# the experiment should accept a design as input and
# produce the qois as output
# experiment(design) = qois

epsInc = 0.02 # strain increment per load step
nInc = 100 # number of strain increments
# qois: \sigma_11, \sigma_22, \eps_33
calc_qois = 3 # number of qois to calculate
qoi_index = [0, 1] # qois to use in calibration
exp_noise_var = [10]

nobs_pstep = [3] #number of observations per step

obs_inds_by_step = [None for step in range(nsteps)]
for ss in np.arange(nsteps):
    first_ob = ss * (nInc - 1)
    last_ob = (ss + 1) * nInc - (ss + 1)
    obs_inds_by_step[ss] = np.round(\
        np.linspace(first_ob, last_ob, nobs_pstep[0]+1, dtype=int)[1:]).tolist()
obs_inds = np.asarray([item for sublist in obs_inds_by_step for item in sublist])

# we are using the model evaluated at the true parameter
# values with added i.i.d. gaussian noise as the experiment

# define the experiment
design_options = ['d1','d2']
experiment = partial(qm.mps_experiment,
    theta_true=theta_true,
    obs_inds=obs_inds,
    calc_qois=calc_qois,
    qoi_index=qoi_index,
    epsInc=epsInc,
    nInc=nInc,
    noise_var=exp_noise_var[0],
    scaler=par_scaler)

test_experiment = experiment(['d1', 'd1'])
# Setting up the forward model:
# fmodel should be a function of the parameters as well as
# the experiment design and provide the qois as output.
# fmodel(theta, design) = qois

model = 'surrogate' # options: 'mps' or 'surrogate'
nobjectives = 1
use_objectives = [0]
if model == 'mps':
    epsInc = 0.02 # strain increment per load step
    nInc = 100 # number of strain increments
    # qois: \sigma_11, \sigma_22, \eps_33
    calc_qois = 3 # number of qois to calculate
    qoi_index = [0, 1] # qois to use in calibration

    nobs_pstep = [3] # number of observations per step
    obs_inds_by_step = [None for step in range(nsteps)]
    for ss in np.arange(nsteps):
        first_ob = ss * (nInc - 1)
        last_ob = (ss + 1) * nInc - (ss + 1)
        obs_inds_by_step[ss] = np.round(\
            np.linspace(first_ob, last_ob, nobs_pstep[0]+1, dtype=int)[1:]).tolist()
    obs_inds = np.asarray([item for sublist in obs_inds_by_step for item in sublist])

    fmodel = partial(\
        qm.mps_qois_fun,
        obs_inds=obs_inds,
        calc_qois=calc_qois,
        qoi_index=qoi_index,
        epsInc=epsInc,
        nInc=nInc,
        return_steps='all',
        scaler=par_scaler)

    fprime = None
    testing_mps_model = fmodel(theta_true, ['d1', 'd2'], return_steps='all')

if model == 'surrogate':
    build_data_dir = "surrogate_data/"
#    surrogates = pickle.load(open(build_data_dir +\
#        f"GP_all_paths_alpha_1e-8_reformatted", "rb"))
#    surrogates = pickle.load(open(build_data_dir +\
#        f"GP_all_paths_alpha_1e-8_7_levels_5_pars_3_obs_pstep_0.02_strainInc", "rb"))
    surrogates = pickle.load(open("GP_all_paths_alpha_1e-8_7_levels_5_pars_3_obs_pstep_0.02_strainInc", "rb"))
    input_scaler = surrogates['input_scaler']
    GP_build_pts = surrogates['xs']
    fmodel = partial(\
        qm.GP_qois_fun_orig,
        surrogates=surrogates,
        input_scaler=input_scaler,
        scale=True,
        return_std=False,
        return_steps='last')

    testing_surr_model = fmodel(\
        theta_true, design=['d1', 'd2', 'd1', 'd2', 'd1', 'd2', 'd1'], return_steps='all')
    mps_compare = qm.mps_qois_fun(\
        samples=theta_true,
        design=['d1', 'd2', 'd1', 'd2', 'd1', 'd2', 'd1'],
        obs_inds=obs_inds,
        num_qois=calc_qois,
        qoi_index=qoi_index,
        epsInc=epsInc,
        nInc=nInc,
        return_steps='all',
        scaler=par_scaler)

    fprime = None

# if posterior method is MCMC
nsamples = 110000
burnin = 10000
cov_check = 2000
target_accept = np.array([0.2, 0.5])
sampler = 'metropolis_hastings'
prop_var = .01
diagnostic_check = nsamples
iter_start_diag = 10000
assert(nsamples > N)

post_update_settings = {}
post_update_settings['MCMC'] = {}
post_update_settings['MCMC']['nsamples'] = nsamples
post_update_settings['MCMC']['burnin'] = burnin
post_update_settings['MCMC']['cov_check'] = cov_check
post_update_settings['MCMC']['target_accept'] = target_accept
post_update_settings['MCMC']['sampler'] = sampler
post_update_settings['MCMC']['prop_var'] = prop_var
post_update_settings['MCMC']['diagnostic_check'] = diagnostic_check
post_update_settings['MCMC']['iter_start_diag'] = iter_start_diag

# if posterior method is Laplace
fd_eps = 'auto' # used in Lapalce approx of evidence
use_own_fd = True
factr=1e1
nrestarts = [3, 10]
pgtol = 1e-7
maxls = 50
fd_order = 4
psd_check_methods = ['scipy', 'eigvals']
post_update_settings['Laplace'] = {}
post_update_settings['Laplace']['fprime'] = fprime
post_update_settings['Laplace']['factr'] = factr
post_update_settings['Laplace']['fd_eps'] = fd_eps
post_update_settings['Laplace']['use_own_fd'] = use_own_fd
post_update_settings['Laplace']['nrestarts'] = nrestarts
post_update_settings['Laplace']['psd_check_methods'] = psd_check_methods
post_update_settings['Laplace']['fd_order'] = fd_order
post_update_settings['Laplace']['maxls'] = maxls
post_update_settings['Laplace']['pgtol'] = pgtol

# posterior updating
posterior_method = 'Laplace' # options: 'Laplace','MCMC'
sequential_design = True # flag for prior to be updated to posterior

# number of times to run boed algorithm to establish reproducibility
ntrials = 1
test_path = [None]*ntrials
test_EIG  = [None]*ntrials
test_expected_val = [None]*ntrials
test_cov = [None]*ntrials
test_avg_time = [None]*ntrials
test_post = [None]*ntrials
test_obs = [None]*ntrials
test_Laplace_summary = [None]*ntrials

workdir = os.getcwd()
par_val = [None] * D
for p, par in enumerate(theta_names):
    par_val[p] = par + str(theta_true[p])
par_ID = '_'.join(par_val)

runID = f'{design_mode}_{posterior_method}_{exp_noise_var[0]}_{nsteps}_{nobs_pstep[0]}_{epsInc}_{model}_{par_ID}'
if design_mode == 'static':
    runID = f'{design_mode}_{posterior_method}_{exp_noise_var[0]}_{nsteps}_{nobs_pstep[0]}_{epsInc}_{model}_{design_initiation[0]}_{par_ID}'
figdir = f"{workdir}/boed_results/figures/{runID}"
pickle_dir = f"{workdir}/boed_results/pickle_files"

figdir_exists = os.path.isdir(figdir)
pickle_dir_exists = os.path.isdir(pickle_dir)
if not figdir_exists:
    os.makedirs(figdir)
if not pickle_dir_exists:
    os.makedirs(pickle_dir)

print("Running Trials")
for test in tqdm(range(ntrials), desc="Processing"):
    if test == 1:
        pause_here = 1

    test_start = time.time()
    boed_dict = run_boed(\
        fmodel,
        experiment,
        nobs_pstep,
        theta_info,
        nsteps=nsteps,
        num_qois=[len(qoi_index)],
        design_options=['d1','d2'],
        design_initiation=design_initiation,
        post_update_settings=post_update_settings,
        noise_var=exp_noise_var,
        fd_eps=fd_eps,
        posterior_method=posterior_method,
        fprime=fprime,
        continuous_design=True,
        nested_mc_settings=nested_mc_settings,
        update_posterior=True,
        figdir=figdir,
        obs_inds=obs_inds,
        nInc=nInc,
        nobjectives=nobjectives,
        use_objectives=use_objectives,
        verbose=True)
        #fd_eps used in finite difference approximations of gradient and hessian


    test_end = time.time()
    test_total = test_end - test_start
    test_path[test] = boed_dict['Design']
    test_EIG[test] = boed_dict['EIG']
    test_post[test] = boed_dict['Post']
    test_expected_val[test] = boed_dict['Post Mean']
    test_cov[test] = boed_dict['Cov']
    test_avg_time[test] = test_total/(nsteps)
    test_obs[test] = boed_dict['Data']
    test_Laplace_summary[test] = boed_dict['Laplace Summary']

    test_data = {}
    test_data['Data'] = test_obs
    test_data['Design'] = test_path
    test_data['EIG'] = test_EIG
    test_data['Post Mean'] = test_expected_val
    test_data['Post Cov'] = test_cov
    test_data['Post'] = test_post
    test_data['avg_time'] = test_avg_time
    test_data['priors'] = theta_priors
    test_data['fmodel'] = fmodel
    test_data['Bounds'] = theta_bounds
    test_data['experiment'] = experiment
    test_data['random seed'] = 1
    test_data['model type'] = model
    test_data['Laplace Summary'] = test_Laplace_summary
    test_data['psd_check_methods'] = psd_check_methods
    test_data['fd_order'] = fd_order
    test_data['pgtol'] = pgtol
    test_data['maxls'] = maxls
    test_data['pickle_dir'] = pickle_dir
    test_data['runID'] = runID
    test_data['factr'] = factr

    if posterior_method == 'MCMC':
        test_data['MCMC Settings'] = post_update_settings['MCMC']
    for item in boed_dict.keys():
        if item not in test_data.keys():
            test_data[item] = boed_dict[item]

    output_name = f'{pickle_dir}/{runID}.pkl'
#    with open(output_name, 'wb') as f:
#        pickle.dump(test_data, f)
print("Complete!")


