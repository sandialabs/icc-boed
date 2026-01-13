import numpy as np
from posterior_update_tools import MCMC
import pickle
from qoi_models import load_cruciform_surrogate, gen_pca_dict
from functools import partial
from build_surrogate_tree import build_tree
import os
import seaborn as sns; sns.set()
import pandas as pd
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.transforms import Affine2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, LogLocator, LinearLocator
from matplotlib import ticker, colormaps
from matplotlib.axis import Axis
import matplotlib
from matplotlib.lines import Line2D
from scipy.stats import norm, multivariate_normal

@ticker.FuncFormatter
def major_formatter(x, pos):
    return f'{x: .2f}'


def get_marginals(pdf):

    D = len(pdf.mean)
    pdf_mean = pdf.mean
    pdf_cov = pdf.cov

    marginals = [None] * D
    for m in range(D):
        marginals[m] = norm(pdf_mean[m], pdf_cov[m,m]**0.5)

    return marginals


def get_joint_marginal(pdf, par_indices):

    D = len(par_indices)
    pdf_mean = pdf.mean
    pdf_cov = pdf.cov
    joint_mean = np.asarray([pdf_mean[i] for i in par_indices])
    joint_cov = np.zeros((D,D))
    for ii in np.arange(D):
      for jj in np.arange(D):
          joint_cov[ii,jj] = pdf_cov[par_indices[ii]][par_indices[jj]]
    joint_marginal = multivariate_normal(joint_mean, joint_cov)

    return joint_marginal


def gen_tree_dict(nlevels, choice_list):

    tree = build_tree(5, choice_list)
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
            path_indice = true_indices[0]
            tree_dict[load_path] = full_paths[path_indice]
    return tree_dict

design = 'AAAAA'
do_analysis = False
make_plots = True
add_laplace = False

filename = f'boed_results/pickle_files/static_{design}_Laplace_Y42.507_A13.6344_b14.3488_a11.1916_pca_modes.pkl'
with open(filename, 'rb') as f:
    picklefile = pickle.load(f)
save_figdir = f'boed_results/figures/static_MCMC_{design}'
save_pickledir = f'boed_results/pickle_files'
if not os.path.exists(save_figdir):
    os.makedirs(save_figdir)
theta_true = picklefile['true par values'].squeeze()

if do_analysis:
    surrogate_dir = 'cruciform_surrogates/surrogates/gps/hosford/5_load_steps/'
    data = picklefile['Data'][0]
    prior = picklefile['priors']
    fmodel = picklefile['fmodel']
    noise_var = picklefile['Noise variance']
    theta_bounds = [[32., 50.], [1., 20.], [0.5, 20.], [4., 16.]]

    nlevels = 5
    choice_list = ['A', 'B']
    tree_dict = gen_tree_dict(nlevels, choice_list)
    gen_pca_dict(tree_dict, surrogate_dir, 'gps', 5)

    nsamples = 120_000
    burnin = 20_000
    cov_check = 2_000
    diagnostic_check = 20_000
    iter_start_diag = 20_000

    design_list = [i for i in design]
    surr = load_cruciform_surrogate(design_list, 'gps', return_steps='all',
        surrogate_dir=surrogate_dir)
    model = partial(fmodel, surr=surr)

    post, _  = MCMC(data, prior, model, noise_var=noise_var, nsamples=nsamples, burnin=burnin,
        cov_check=cov_check, diagnostic_check=diagnostic_check, return_pca=True,
        use_objectives=[0, 1], num_qois=[2,2], design=design_list,
        bounds=theta_bounds, nobjectives=2, exemplar='cruciform',
        figdir=save_figdir, iter_start_diag=iter_start_diag)

    with open(f'{save_pickledir}/static_MCMC_{design}.pkl', 'wb') as f:
        pickle.dump(post, f)

if make_plots:
    with open(f'{save_pickledir}/static_MCMC_{design}.pkl', 'rb') as f:
        mcmc_analysis = pickle.load(f)

    if add_laplace:
        mcmc_alpha = 0.2
    else:
        mcmc_alpha = 1
    burnin = 20000
    thin = 10
    fontsize = 18
    mathsize = 20
    sns.set_style("ticks")
    if design == 'AAAAA':
        cmap = 'viridis'
        color = 'seagreen'
    elif design == 'ABABA':
        cmap = 'plasma'
        color = 'blueviolet'

    trace = mcmc_analysis['theta']
    par_names = ['Y', 'A', 'n', 'a' ]
    D = len(par_names)
    par_df = pd.DataFrame(trace, columns=par_names)

    fig, ax = plt.subplots(D, D, figsize=(12,12))
    for col in range(D):
        for row in range(D):
            pars = [par_names[row], par_names[col]]
            sub_df = par_df[pars][burnin::thin]

            if row == col:

                extend_yval = False
                xmin = np.percentile(sub_df[pars[0]], [1])
                if theta_true[row] < xmin:
                    xmin = .95 * theta_true[row]
                    extend_yval = True
                xmax = np.percentile(sub_df[pars[0]], [99])
                if theta_true[row] > xmax:
                    xmax = 1.05 * theta_true[row]
                    extend_yval = True

                cred_int = np.percentile(sub_df[pars[0]], [2.5, 97.5])
                # kde of marginal posterior
                max_val = sub_df[pars[0]].max()[0]
                kde = sns.kdeplot(par_df.loc[burnin::thin, pars[0]],
                    linewidth=4, fill=False,
                    ax=ax[row,col], color=color)
                if not add_laplace:
                    line = kde.lines[0]
                    yval = line.get_data()[1].max()
                    ax[row, col].plot([cred_int[0], cred_int[0]],
                        [0, yval], color='black', linestyle='--', linewidth=3)
                    ax[row, col].plot([cred_int[1], cred_int[1]],
                        [0, yval], color='black', linestyle='--', linewidth=3)
                    ax[row, col].plot([theta_true[row], theta_true[row]],
                        [0, yval], color='magenta', linestyle='--', linewidth=3)
                ax[row,col].grid(False)
                ax[row,col].set_facecolor('white')
                ax[row, col].set_ylabel('')
                ax[row, col].set_xlabel('')
                ax[row, col].set_yticks([])
                col_ticks = cred_int #ax[row,col].get_xticks()
                if col < D - 1:
                    ax[row, col].set_xticks([])
                ax[row, col].set_ylim(bottom=0)

            if row > col:

            # kde of joint-posterior
                sns.kdeplot(data=par_df.loc[burnin::thin,pars], levels=100,
                    fill=False, cmap=cmap, x=pars[1] , y=pars[0],
                    ax=ax[row,col], thresh=0.025, alpha=mcmc_alpha)
                ax[row, col].plot(theta_true[col], theta_true[row], color='magenta', marker='*', ms=20)
                ax[row, col].grid(False)
                ax[row, col].set_facecolor('white')
                ax[row, col].set_xticks([])
                ax[row, col].set_ylabel('')
                ax[row, col].set_xlabel('')

                if col > 0:
                    ax[row, col].set_yticks([])

            if row == D-1:
                ax[row, col].set_xticks([col_ticks[1], col_ticks[-2]])
                ax[row, col].tick_params(axis='x', labelsize=fontsize)
                ax[row, col].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
                if col == 0:
                    ax[row, col].set_xlabel(r"${0} \quad \mathrm{{(ksi)}}$".format("\sigma_{y}"), fontsize=mathsize, fontweight='bold')
                elif col == 1:
                    ax[row, col].set_xlabel(r"${0} \quad \mathrm{{(ksi)}}$".format(par_names[col]), fontsize=mathsize, fontweight='bold')
                else:
                    ax[row, col].set_xlabel(r"${0}$".format(par_names[col]), fontsize=mathsize, fontweight='bold')

            if col == 0 and row > 0:
                if row == 1:
                    ax[row, col].set_ylabel(r"${0} \quad \mathrm{{(ksi)}}$".format(par_names[row]), fontsize=mathsize, fontweight='bold')
                else:
                    ax[row, col].set_ylabel(r"${0}$".format(par_names[row]), fontsize=mathsize, fontweight='bold')

#                    for key, spine in ax[row, col].spines.items():
#                        spine.set_visible(True)
            if col == 0:
                ax[row, col].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))

            if row < col:
                fig.delaxes(ax[row, col])

    for row in np.arange(1,D):
        ticks = ax[D-1, row].get_xticks()
        ax[row, 0].set_yticks(ticks)
        ax[row, 0].tick_params(axis='y', labelsize=fontsize)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    if not add_laplace:
        plt.savefig(f'{save_figdir}/MCMC_contours_{design}.png')
        plt.savefig(f'{save_figdir}/MCMC_contours_{design}.pdf')


if add_laplace:
    with open(f'{save_pickledir}/static_MCMC_{design}.pkl', 'rb') as f:
        mcmc_analysis = pickle.load(f)
    Laplace_filename = f"static_{design}_Laplace_Y42.507_A13.6344_b14.3488_a11.1916_pca_modes.pkl"
    workdir = os.getcwd()
    filedir = f"{workdir}/boed_results/pickle_files/"
    full_path_to_file = f"{filedir}/{Laplace_filename}"
    file = open(full_path_to_file, 'rb')
    pkl_file = pickle.load(file)
    post = pkl_file['Post'][0][-1]
    marginal_post = get_marginals(post)
    gridsize = [0.001, 0.001, .001, .001]
    alpha_val = 0.3
    pdf = post
    marginals = marginal_post

    for col in range(D):
        for row in range(D):
            if row == col:
                xmin = marginal_post[row].ppf(.0001)
                if theta_true[row] < xmin:
                    xmin = .95 * theta_true[row]
                    extend_yval = True
                xmax = marginal_post[row].ppf(.9999)
                if theta_true[row] > xmax:
                    xmax = 1.05 * theta_true[row]
                    extend_yval = True

                xgrid = np.linspace(xmin, xmax, 1000)
                # hist of marginal posterior
                max_val = marginals[row].pdf(xgrid).max()
                hist_plot  = ax[row,col].plot(xgrid, marginals[row].pdf(xgrid)/max_val, color='black', linewidth=4)
                ipf=interp1d(x=xgrid, y=marginals[row].pdf(xgrid))
                quantile_025 = marginals[row].ppf(.025)
                quantile_975 = marginals[row].ppf(.975)
                yval = ipf(theta_true[row])
                if extend_yval or yval < 1:
                    yval = 0.5 * marginals[row].pdf(xgrid).max()
                yval = 1 #marginals[row].pdf(xgrid).max()
                ax[row, col].plot([theta_true[row], theta_true[row]],
                    [0, yval], color='magenta', linestyle='--', linewidth=3)

            if row > col:
                    par_indices = [col, row]
                    joint_pdf = get_joint_marginal(pdf, par_indices)
                    x,y = np.mgrid[marginals[col].ppf(.0009): marginals[col].ppf(.9999):gridsize[col],
                        marginals[row].ppf(.0009):marginals[row].ppf(.9999):gridsize[row]]
                    pos = np.dstack((x,y))
                    min_level = joint_pdf.pdf(pos).min()
                    max_level = joint_pdf.pdf(pos).max()
                    levels = np.linspace(min_level, max_level, 200)
                    ax[row, col].contour(x, y, joint_pdf.pdf(pos), levels=200, colors='black', alpha=alpha_val)
                    ax[row, col].plot(theta_true[col], theta_true[row], color='magenta', marker='*', ms=20)

    plt.savefig(f'{save_figdir}/Laplace_MCMC_contours_{design}.png')
    plt.savefig(f'{save_figdir}/Laplace_MCMC_contours_{design}.pdf')

