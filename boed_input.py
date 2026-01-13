import numpy as np
import matplotlib.pyplot as plt

import pickle
import random
import os, sys
import time

from functools import partial
from scipy.stats import norm, uniform, multivariate_normal, invgamma
from tqdm import tqdm

# my modules
import qoi_models
import stats_utils as stu
import warnings
import utils
from qoi_models import MTS_standin

from boed_alg import run_boed
from build_surrogate_tree import build_tree
from evaluate_models import compute_qois
from posterior_update_tools import Laplace_approx, MCMC
import matplotlib.pyplot as plt
import matplotlib

sys.path.insert(0, 'paper_figures_and_scripts')

def gen_tree_dict(nlevels, choice_list):

    tree = build_tree(nlevels, choice_list)
    ntree_nodes = 0
    for tree_level in tree:
        ntree_nodes += len(tree_level)
    paths = [None] * ntree_nodes

    path_idx = 0
    for level in np.arange(nlevels):
        for path in tree[level]:
            paths[path_idx] = list(path)
            path_idx += 1
    paths = [i for i in paths if i is not None]

    full_paths = ["".join(list(item)) for item in tree[-1]]
    tree_dict = {}
    for p, path in enumerate(paths):
        load_path = ''.join(path)
        if load_path.startswith('A'):
            # determine which load path to pull data from
            corr_full_paths = [fp.startswith(load_path) for fp in full_paths]
            true_indices = [i for i, x in enumerate(corr_full_paths) if x]
            save_indices = [idx for idx in true_indices if full_paths[idx].startswith('AB')]
            try:
                path_indice = save_indices[0]
            except:
                continue
            #path_indice = true_indices[0]
            tree_dict[load_path] = full_paths[path_indice]
        if load_path == 'A':
            # determine which load path to pull data from
            if nlevels == 6:
                tree_dict[load_path] = 'ABAAAA'
            elif nlevels == 5:
                tree_dict[load_path] = 'ABAAA'
    return tree_dict


# set random seed
np.random.seed(1)

# set up directory for saving figures
workdir = os.getcwd()
dirname = f"../data/boed_results/cruciform_figures"
figdir_exists = os.path.isdir(dirname)
if not figdir_exists:
    os.makedirs(dirname)

# set path for pre-built surrogates (depends on surrogate type, material model and specific case)
surrogate_type = 'gps' #'pce' or 'gps'
mat_model = 'hosford'
model_case = '6_load_steps'
model_id = f'{mat_model}/{model_case}'
surrogates_dir = f"cruciform_surrogates/surrogates/{surrogate_type}/{model_id}/new_bcs"
nmodes = 5

#####################################
######### BOED settings #############
#####################################

# number of BOED steps (each steps consists of an EIG calculation, data collection, and calibration)
nsteps = 2

# Nested MC Settings: set the number of samples in outer (N) and inner (M)
# loop of Nested MC EIG approximation
M = 10 ** 2
N = 10 ** 2

# State whether to standardize the experimental data
standardize = False

ind_inner_samples = False # If false, the inner samples are the same for each outer loop in N. If true, the inner samples are unique for each outer loop N.
ind_inner_outer_samples = True # If False: the M inner samples are the same as the N outer samples.

# define the method to be used in the inner loop (evidence term) calculation of the EIG
evidence_method = 'importance_sampling' # options: 'Laplace', 'importance_sampling'
biasing_dist = 'prior' #options: 'prior', 'laplace_based', 'self_normalizing', 'self_centered'
nested_mc_settings = {}
nested_mc_settings['M'] = M
nested_mc_settings['N'] = N
nested_mc_settings['ind_inner_samples'] = ind_inner_samples
nested_mc_settings['ind_inner_outer_samples'] = ind_inner_outer_samples
nested_mc_settings['evidence_method'] = evidence_method
nested_mc_settings['biasing_dist'] = biasing_dist


# If len(design_initiation) == nsteps, this produces a static design
# EIG is not calculated since experiment is predetermined at every step
design_initiation = ['A', 'B']

if len(design_initiation) < nsteps:
    design_mode = 'adaptive'
elif len(design_initiation) == nsteps:
    design_mode = 'static'
elif len(design_initiation) > nsteps:
    nsteps = len(design_initiation)
    design_mode = 'static'

# Generate load step tree
nlevels = 6
design_options = ['A','B']
tree_dict = gen_tree_dict(nlevels, design_options)

######################################
##### Synthetic Data Generation #####
######################################

# location of surrogate data (used to generate synthetic data)
build_data_dir = f"cruciform_surrogates/surrogate_data/{model_id}/new_bcs/"

# If using synthetic experimental, the data is pulled from
# the surrogate test data generated with parameters with
# lowest percent difference from  values given by Kyle
ntrain_samples = 500
features = np.load(build_data_dir + "samples.npy")
if mat_model == 'hosford':
    true_vals = np.array((42.2, 12.8, 14.5, 8.0)) # from kyle: 42 in ksi
    abs_diff = np.abs(features - true_vals)
    tot_diff = abs_diff.sum(axis=1)
elif mat_model == 'hill':
    if model_case == 'five_param':
        true_vals = np.array((42.2, 12.8, 14.5, 1.07, 0.84))
    elif model_case == 'six_param':
        true_vals = np.array((42.2, 12.8, 14.5, 1.07, 0.84, 0.89))
    abs_perc_diff = np.abs(features - true_vals)/np.abs(true_vals)
    tot_diff = abs_perc_diff[:, -3:].sum(axis=1)
sorted_diff = np.argsort(list(tot_diff))
sorted_diff_test = [i for i in sorted_diff if i > ntrain_samples]
experiment_sample_number = sorted_diff_test[0]
theta_true = features[experiment_sample_number, :].round(4)

######################################
##### Prior modeling assumptions #####
######################################

print(f"theta true: {theta_true}")
if mat_model == 'hosford':
    theta_bounds = [[32., 50.], [1., 20.], [0.5, 20.], [4., 16.]]
    prior_mean = [40.0, 10.0, 10.0, 10.0]
    prior_std = [15, 15, 15, 15]
    theta_names = ["Y", "A", "b", "a"]
elif mat_model == 'hill':
    if model_case == 'five_param':
        theta_bounds = [[32., 50.], [1., 20.], [0.5, 20.],
            [0.85, 1.15], [0.85, 1.15]]
        prior_mean = [40.0, 10.0, 10.0, 1.0, 1.0]
        prior_std = [15, 15, 15, 1, 1]
        theta_names = ["Y", "A", "b", "R12", "R22"]
    elif model_case == 'six_param':
        theta_bounds = [[30., 50.], [1., 20.], [5.0, 50.],
            [0.85, 1.15], [0.85, 1.15], [0.85, 1.15]]
        prior_mean = [40.0, 10.0, 10.0, 1.0, 1.0, 1.0]
        prior_std = [15, 15, 15, 1, 1, 1]
        theta_names = ["Y", "A", "b", "R12", "R22", "R33"]
D = len(theta_names)

prior_dist = ['trunc_normal'] * D
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
    if not utils.isPD(pr_var):
        pr_var = utils.nearest_PD(pr_var)

    det_tol = 1e-10
    if np.linalg.det(pr_var) < det_tol:
        while np.linalg.det(pr_var) < det_tol:
            pr_var += np.eye(D) * 1e-10

    try:
        theta_priors = [multivariate_normal(\
            pr_mean, pr_var)]
    except:
         if utils.isPD(pr_var):
             theta_priors = [\
                 multivariate_normal(\
                 pr_mean, pr_var, allow_singular=True)]
         else:
             raise ValueError('Prior covariance not PD')


theta_0 = prior_mean
theta_info = {}
theta_info['true'] = theta_true
theta_info['bounds'] = theta_bounds
theta_info['prior'] = theta_priors
theta_info['init'] = theta_0
theta_info['names'] = theta_names

######################################
####### Set up forward model #########
######################################
nobjectives = 2 #number of objectives model can produce
use_objectives = [0, 1] #objective to use in calibration
num_qois = [2, 2] # number of qois to calculate
qoi_index = [[0, 1], [0, 1]] # qois to use in calibration
integrate_displacement = False
if integrate_displacement:
    qoi_models.gen_mass_matrix()
return_pca = False
if integrate_displacement and return_pca:
    raise ValueError('integrate_displacement and return_pca cannot both be True.')

fmodel = partial(
    qoi_models.cruciform_surr_qois_fun,
    qoi_index=qoi_index,
    integrate_displacement=integrate_displacement,
    return_pca=return_pca,
    use_objectives=use_objectives
)

surr = qoi_models.load_cruciform_surrogate(['A','B'],
    surrogate_type, return_steps='last', surrogate_dir=surrogates_dir)
test_model = fmodel(theta_true, surr=surr)
fprime = None

##################################
##### Set up experiment ##########
##################################
exp_noise_var = np.array([4e-12, 582.11]) # load std ~.05% error of the reading at final load step (reported in Lbs)
if integrate_displacement:
    nobs_pstep = [1, 1]
    data_space = 'integrated_disp'
elif return_pca:
    nobs_pstep = [nmodes, 1]
    data_space = 'pca_modes'
else:
    nobs_pstep = [1022, 1]
    data_space= 'field_data'

# generate tree for looking up pca basis, mean and scalers
# for each load step and direcitonal component
qoi_models.gen_pca_dict(tree_dict, surrogates_dir, surrogate_type, nmodes)


#experiment = partial(MTS_standin, return_pca=return_pca, use_objectives=use_objectives)
experiment = partial(qoi_models.cruciform_experiment,
    experiment_sample_number=experiment_sample_number,
    qoi_index=qoi_index,
    tree_dict=tree_dict,
    build_data_dir=build_data_dir,
    noise_var=exp_noise_var,
    integrate_displacement=integrate_displacement,
    return_pca=return_pca,
    use_objectives=use_objectives,
    theta_true=theta_true)


###############################################
###### Posterior update  user definitions #####
###############################################
posterior_method = 'MCMC' # options: 'Laplace','MCMC'
sequential_design = True # flag for prior to be updated to posterior

nsamples = 2000 #11000000
burnin = 500 #10000
cov_check = 500 #1000
target_accept = np.array([0.2, 0.5])
sampler = 'metropolis_hastings'
prop_var = .01
diagnostic_check = 1000 #10000
iter_start_diag = 500 #10000
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
nrestarts = [1,5]
pgtol = 1e-7
maxls = 50
fd_order = 4
psd_check_methods = ['all']
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
test_gen_var = [None]*ntrials
test_tot_var = [None]*ntrials
test_field_data = [None]*ntrials

workdir = os.getcwd()
par_val = [None] * D
for p, par in enumerate(theta_names):
    par_val[p] = par + str(theta_true[p])
par_ID = '_'.join(par_val)

if design_mode == 'static':
    static_design = ''.join(design_initiation)
    runID = "<insert runID>"
des_str = ''.join(design_initiation)

figdir = f"<example_figdir>"
pickle_dir = f"../data/boed_results/pickle_files"

figdir_exists = os.path.isdir(figdir)
pickle_dir_exists = os.path.isdir(pickle_dir)
if not figdir_exists:
    os.makedirs(figdir)


boed_dict = run_boed(\
    fmodel, experiment, nobs_pstep, theta_info,
    nsteps=nsteps,  num_qois=num_qois,
    design_options=design_options, design_initiation=design_initiation,
    utility_function='EIG', likelihood='normal',
    posterior_method=posterior_method,  post_update_settings=post_update_settings,
    sequential_design=True, noise_var=exp_noise_var, fd_eps=fd_eps, fprime=fprime,
    continuous_design=True, nested_mc_settings=nested_mc_settings,
    update_posterior=True, figdir=figdir, nobjectives=nobjectives,
    return_pca=return_pca, verbose=True, use_objectives=use_objectives,
    surrogate_type=surrogate_type,
    surrogate_dir=surrogates_dir,
    exemplar='cruciform',
    standardize=standardize,
    integrate_displacement=integrate_displacement)

    # continuous_design: the outcome of the nth experiment depends
    # on the outcome of experiments 1:(n-1)

    # sequential_design: the posterior becomes the sampling distribution
    # (prior) for the EIG calculation in the next step

output_name = "<insert output name>"
#with open(output_name, 'wb') as f:
#    pickle.dump(boed_dict, f)

print("Complete!")
