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

# set up directory for saving images
workdir = os.getcwd()
dirname = f"../data/boed_results/cruciform_figures"
figdir_exists = os.path.isdir(dirname)
if not figdir_exists:
    os.makedirs(dirname)

surrogate_type = 'gps' # option: 'pce' or 'gps'
mat_model = 'hill'
model_case = 'six_param'
model_id = f'{mat_model}/{model_case}'
surrogates_dir = f"cruciform_surrogates/surrogates/{surrogate_type}/{model_id}/extended_bounds"
#surrogates_dir = f"cruciform_surrogates/cruciform_{surrogate_type}/hill/six_param/"
nmodes = 5

build_data_dir = f"cruciform_surrogates/surrogate_data/{model_id}/extended_bounds/"
#build_data_dir = "cruciform_surrogates/surrogate_data/AB_start_6load_steps/extended_cruciform_surrogate_data/"
#build_data_dir = "cruciform_surrogates/surrogate_data/hill/six_param/"

#####################################
######### BOED settings #############
#####################################

nsteps = 4

# Nested MC Settings
M = 10 ** 3
N = 10 ** 4

standardize = False

ind_inner_samples = False # independent M samples for each N
ind_inner_outer_samples = True # M and N samples independent of each other

evidence_method = 'importance_sampling' # options: 'Laplace', 'importance_sampling'
biasing_dist = 'prior' #options: None, 'prior', 'laplace_based', 'self-normalizing', 'self_centered'
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

# OED
design_initiation = ['A', 'B', 'B', 'B']

# no OED (if nsteps = 5)
#design_initiation = ["A", "B", "A", "B", "A"]

if len(design_initiation) < nsteps:
    design_mode = 'adaptive'
elif len(design_initiation) == nsteps:
    design_mode = 'static'
elif len(design_initiation) > nsteps:
    nsteps = len(design_initiation)
    design_mode = 'static'

nlevels = 6
choice_list = ['A', 'B']
tree_dict = gen_tree_dict(nlevels, choice_list)

# choose sample that is relatively close to true values
# or use calibration data and true values with static setting
ntrain_samples = 500
if True:
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

#theta_true = np.array((42.507, 13.6344, 14.3488, 11.1916))
#experiment_sample_number = 501

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


if False:
    fix, ax = plt.subplots(2, D, figsize=(16,8))
    for d in np.arange(D):
        min_val = theta_bounds[d][0]*1.01
        max_val = theta_bounds[d][1]*0.99
        xgrid = np.linspace(min_val, max_val, 5000)
        log_yval = np.log(theta_priors[d].pdf(xgrid))
        yval = theta_priors[d].pdf(xgrid)
        ax[0,d].plot(xgrid, yval)
        ax[1,d].plot(xgrid, log_yval)
        plt.suptitle(f"Log Prior for {theta_names[d]}")
    plt.savefig("theta_priors.png")

theta_0 = prior_mean
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

num_qois = [2, 2] # number of qois to calculate
qoi_index = [[0, 1], [0, 1]] # qois to use in calibration
exp_noise_var = np.array([4e-12, 582.11]) # load std ~.05% error of the reading at final load step (reported in Lbs)
#exp_noise_var = np.array([10e-6 ** 2, 55 ** 2]) # load std ~.05% error of the reading at final load step (reported in Lbs)

# define the experiment - use instance of surrogate
# test data not used for training

#qoi_models.gen_mass_matrix()
int_disp = False
if int_disp:
    qoi_models.gen_mass_matrix()

return_pca = True
if int_disp:
    nobs_pstep = [1, 1]
    data_space = 'integrated_disp'
elif return_pca:
    nobs_pstep = [nmodes, 1]
    data_space = 'pca_modes'
else:
    nobs_pstep = [1022, 1]
    data_space= 'field_data'

#if return_pca:
qoi_models.gen_pca_dict(tree_dict, surrogates_dir, surrogate_type, nmodes)

design_options = ['A','B']
nobjectives = 2 #number of objectives model can produce
use_objectives = [0] #objective to use in calibration
if False:
    experiment = partial(qoi_models.cruciform_experiment,
        experiment_sample_number=experiment_sample_number,
        qoi_index=qoi_index,
        tree_dict=tree_dict,
        build_data_dir=build_data_dir,
        noise_var=exp_noise_var,
        integrate_displacement=int_disp,
        return_pca=return_pca,
        use_objectives=use_objectives,
        theta_true=theta_true)

    if False:
        pca_data = {}
        disp_data = {}
        load_data = {}
        for t, branch in enumerate(tree_dict):
            path = [char for char in branch]
            data = experiment(path)
            pca_data[branch] = data[0]
            load_data[branch] = data[1]
            disp_data[branch] = data[2]
        with open('simulated_data.pkl', 'wb') as f:
            pickle.dump([pca_data, load_data, disp_data], f)

#variable_increment = 0.225
#experiment = partial(qoi_models.variable_load_step_experiment,
#    increment=variable_increment,
#    noise_var=exp_noise_var,
#    return_pca=return_pca)


#experiment = partial(qoi_models.abbreviated_cruciform_experiment,
#                     use_objectives=use_objectives)

experiment = partial(MTS_standin, return_pca=return_pca, use_objectives=use_objectives)

# Setting up the forward model:
# fmodel should be a function of the parameters as well as
# the experiment design and provide the qois as output.
# fmodel(theta, design) = qois
model = 'surrogate'
fmodel = partial(
    qoi_models.cruciform_surr_qois_fun,
    qoi_index=qoi_index,
    integrate_displacement=int_disp,
    return_pca=return_pca,
    use_objectives=use_objectives
)

surr = qoi_models.load_cruciform_surrogate(['A','B'],
    surrogate_type, return_steps='last', surrogate_dir=surrogates_dir)
test_model = fmodel(theta_true, surr=surr)
#test_exp = experiment(['A','B'])
fprime = None

# if posterior method is MCMC
nsamples = 11000000
burnin = 10000
cov_check =1000
target_accept = np.array([0.2, 0.5])
sampler = 'metropolis_hastings'
prop_var = .01
diagnostic_check = 10000
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
test_gen_var = [None]*ntrials
test_tot_var = [None]*ntrials
test_field_data = [None]*ntrials

workdir = os.getcwd()
par_val = [None] * D
for p, par in enumerate(theta_names):
    par_val[p] = par + str(theta_true[p])
par_ID = '_'.join(par_val)

#runID = f'{mat_model}_{model_case}_{design_mode}_{posterior_method}_{nsteps}_{par_ID}'
if design_mode == 'static':
    static_design = ''.join(design_initiation)
    runID = f'{mat_model}_{model_case}_{design_mode}_{static_design}_{posterior_method}_{par_ID}'
des_str = ''.join(design_initiation)
runID = f"ST10_hill6_extended_bounds_disp_only"

figdir = f"../data/boed_results/figures/{runID}"
pickle_dir = f"../data/boed_results/pickle_files"

figdir_exists = os.path.isdir(figdir)
pickle_dir_exists = os.path.isdir(pickle_dir)
if not figdir_exists:
    os.makedirs(figdir)

if True:
    from make_boed_figures import get_marginals
    #for p, path in enumerate(tree_dict.keys()):
    #    if len(path) == nsteps:
    #        print('***********************')
    #        print(f"    {path}     ")
    #        print('***********************')
    #        design_initiation = path
    #        static_design = ''.join(design_initiation)
    #        runID = f'static_Laplace_{static_design}'
    pickle_dir = "/home/dericci/local/mc2/files/paper_figures_and_scripts/experimental_paper/pkl_files"
    filename = f"{pickle_dir}/MTS_ST10_field_data.pkl"
    ls = -1
#        filename = f'{pickle_dir}/{runID}_{data_space}.pkl'
    with open(filename, 'rb') as f:
        pdata = pickle.load(f)
    post = pdata['Post']
    cov = post[0][ls].cov
    var = np.diag(cov)
    std = var ** 0.5
    mean = post[0][ls].mean
    marginals = get_marginals(post[0][ls])
    pgen_var = np.linalg.det(cov)
    ptot_var = np.matrix.trace(cov)
    #conversion_factor = [6.8948, 6.8948, 1, 1]
    conversion_factor = [1, 1, 1, 1]
    #diff = theta_true - mean
    #md = np.sqrt(diff.T @ np.linalg.inv(cov) @ diff)
    #print(f"Expected value: {post[0][-1].mean}")
    #print(f"Marginal variances: {np.diag(post[0][-1].cov)}")
    #print(f"gen var: {pgen_var}")
    #print(f"log(gen var): {np.log(pgen_var)}")
    #print(f"tot val: {ptot_var}")
    #print(f"Bias norm: {np.linalg.norm(theta_true - post[0][-1].mean)}")
    #print(f"Mahalanobis distance: {md}")
    for dd in np.arange(4):
        quantile_025 = marginals[dd].ppf(.025) * conversion_factor[dd]
        quantile_975 = marginals[dd].ppf(.975) * conversion_factor[dd]
        plus_minus = (quantile_975 - quantile_025)/2
        print(f"expected value {theta_names[dd]}: {np.round(marginals[dd].mean() * conversion_factor[dd], 3)}")
        print(f"Standard deviation {theta_names[dd]}: {np.round(std[dd] * conversion_factor[dd], 3)}")
        print(f"95 CI {theta_names[dd]}: ({np.round(quantile_025, 2)},{np.round(quantile_975, 2)})")
        print(f"95 CI plus/minus {theta_names[dd]}: {np.round(plus_minus, 2)}")
        expected_value = np.round(marginals[dd].mean(), 4)
        assert np.isclose(expected_value * conversion_factor[dd] - quantile_025, plus_minus, atol=1e-3)
        assert np.isclose(quantile_975 - expected_value * conversion_factor[dd], plus_minus, atol=1e-3)

                #print(f"plus/minus: {plus_minus}")
    assert False

print("Running Trials")
for test in tqdm(range(ntrials), desc="Processing"):
    test_start = time.time()
    #for p, path in enumerate(tree_dict.keys()):
    #    if len(path) == nsteps: # and path == ''.join(design_initiation):
    #    #if path == 'AAAAA':
    #        design_initiation = [i for i in path]
    #        static_design = path
    #        print(f"calibraiton for static design {static_design}")
    #        runID = f'static_Laplace_{static_design}'
    boed_dict = run_boed(\
        fmodel, experiment, nobs_pstep, theta_info,
        nsteps=nsteps,  num_qois=num_qois,
        design_options=['A', 'B'], design_initiation=design_initiation,
        utility_function='EIG', likelihood='normal',
        posterior_method=posterior_method,  post_update_settings=post_update_settings,
        sequential_design=True, noise_var=exp_noise_var, fd_eps=fd_eps, fprime=fprime,
        continuous_design=True, nested_mc_settings=nested_mc_settings,
        update_posterior=True, figdir=figdir, nobjectives=nobjectives,
        return_pca=return_pca, verbose=True, use_objectives=use_objectives,
        surrogate_type=surrogate_type,
        surrogate_dir=surrogates_dir,
        exemplar='cruciform',
        standardize=standardize)

        # continuous_design: the outcome of the nth experiment depends
        # on the outcome of experiments 1:(n-1)

        # sequential_design: the posterior becomes the sampling distribution
        # (prior) for the EIG calculation in the next step

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
#    test_gen_var[test] = [np.linalg.det(test_cov[test][ii]) for ii in np.arange(nsteps)]
#    test_tot_var[test] = [np.matrix.trace(test_cov[test][ii]) for ii in np.arange(nsteps)]
    test_field_data[test] = boed_dict['Field Data']

    test_data = {}
    test_data['Data'] = test_obs
    test_data['Field Data'] = test_field_data
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
    test_data['gen_var'] = test_gen_var
    test_data['tot_var'] = test_tot_var
    test_data['material_model'] = mat_model
    test_data['model_case'] = model_case
    if posterior_method == 'MCMC':
        test_data['MCMC Settings'] = post_update_settings['MCMC']
    for item in boed_dict.keys():
        if item not in test_data.keys():
            test_data[item] = boed_dict[item]

            if posterior_method == 'MCMC':
                test_data['MCMC Settings'] = post_update_settings['MCMC']
            for item in boed_dict.keys():
                if item not in test_data.keys():
                    test_data[item] = boed_dict[item]

    output_name = f'{pickle_dir}/{runID}_{data_space}.pkl'
    post = test_data['Post']
    pgen_var = np.linalg.det(post[0][-1].cov)
    print(f'gen var = {pgen_var}')
    #with open(output_name, 'wb') as f:
    #    pickle.dump(test_data, f)

if False:
    font = {'size': 24}
    matplotlib.rc('font', **font)
    from make_boed_figures import get_marginals
    post = test_data['Post'][0][-1]
    marginals = get_marginals(post)
    fix, axs = plt.subplots(1, D, figsize=(18,6))
    for dd in np.arange(D):
        xmin = marginals[dd].ppf(.0005)
        xmax = marginals[dd].ppf(.9995)
        xgrid = np.linspace(xmin, xmax, 1000)

        quantile_025 = marginals[dd].ppf(.025)
        quantile_975 = marginals[dd].ppf(.975)
        true_val = theta_true[dd]
        pdf = marginals[dd].pdf(xgrid)
        max_val = pdf.max()

        axs[dd].plot(xgrid, pdf/max_val, 'g', linewidth=3, label='pdf')
        #axs[dd].vlines(true_val, 0, 1, 'r', '--', linewidth=4, label='true val')
        axs[dd].vlines(quantile_025, 0, 1, 'k', '--', linewidth=3, label='95% CI')
        axs[dd].vlines(quantile_975, 0, 1, 'k', '--', linewidth=3)
        if dd == D-1:
            axs[dd].legend()
        if dd == 0:
            axs[dd].set_ylabel('pdf')
        axs[dd].set_xlabel(f'{theta_names[dd]}')
    plt.show()
    # optional: do MCMC at the end with all data
    if False:
        data = test_data['Data'][0]
        design = design_initiation
        surr = qoi_models.load_cruciform_surrogate(\
            design, surrogate_type, return_steps='all',
            surrogate_dir=surrogates_dir)
        fmodel_des_fixed = partial(fmodel, surr=surr)

        from utility_functions import draw_samples
        theta_init = draw_samples(theta_priors, D, 1, theta_bounds)
        from posterior_update_tools import MCMC
        trace, prop_var = MCMC(data, theta_priors, fmodel_des_fixed,
            nsamples, burnin, cov_check, sampler, prop_var,
            theta_bounds, theta_init, exp_noise_var, 'normal',
            diagnostic_check, iter_start_diag, theta_true,
            figdir, 4, nobjectives=nobjectives,
            use_objectives=use_objectives,
            return_pca=return_pca, design=design,
            num_qois=num_qois)

print("Complete!")
