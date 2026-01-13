import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib import ticker
import matplotlib as mpl
import sys
import os
import pandas as pd
import seaborn as sns; sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})

def diagnostics(trace, k, figdir=None, boe_step=None, theta_true=None, iter_start_diag=1):

    if boe_step is None:
        boe_step = 1

    # parameter trace plot
    D = trace["theta"].shape[1]

    if D == 4:
        par_names = {'Y', 'A', 'b', 'a'}
    elif D == 5:
        par_names = {'Y', 'A', 'b', 'R12', 'R22'}
    elif D == 6:
        par_names = {'Y', 'A', 'b', 'R12', 'R22', 'R33'}

    fig, ax = plt.subplots(1,D,figsize=(4*D, 6))
    if D > 1:
        for d in range(D):
            ax[d].plot(trace["theta"][iter_start_diag:k,d])
            if theta_true is not None:
                ax[d].axhline(y=theta_true[0][d], color='k',\
                    linewidth=2, linestyle='--', xmin=0, xmax=1)
    else:
        ax.plot(trace["theta"][iter_star_diag:k])
        if theta_true is not None:
            ax.axhline(y=theta_true, color='k',\
                linewidth=2, linestyle='--', xmin=0, xmax=1)

    plt.savefig(f"{figdir}/parameter_traces_step_{boe_step}_iter_{k}.png")
    plt.savefig(f"{figdir}/parameter_traces_step_{boe_step}_iter{k}.pdf")
    plt.close("all")


    # log post trace plot
    fig, ax = plt.subplots(1,1,figsize=(12,8))
    ax.plot(trace["log_post"][iter_start_diag:k])
    plt.savefig(f"{figdir}/logpost_trac_step_{boe_step}_iter_{k}.png")
    plt.savefig(f"{figdir}/logpost_trace_step_{boe_step}_iter_{k}.pdf")
    plt.close("all")


    # parameter correlation plot
    trace_df = pd.DataFrame(trace["theta"][iter_start_diag:k,:], columns=par_names)
    pd.plotting.scatter_matrix(trace_df, alpha=0.2)
    plt.savefig(f"{figdir}/corr_step_{boe_step}_iter_{k}.png")
    plt.savefig(f"{figdir}/corr_step_{boe_step}_iter_{k}.pdf")
    plt.close("all")


    # kde plots
    if False:
        grid = sns.PairGrid(trace_df, diag_sharey=False, corner=True)
        grid = grid.map_lower(sns.kdeplot, levels=100,
            thresh=0, fill=True, cmap="mako")
        grid = grid.map_diag(sns.kdeplot, thresh=0.5)
        ax.grid(False)

        plt.savefig(f"{figdir}/kde_step_{boe_step}_iter_{k}.png")
        plt.savefig(f"{figdir}/kde_step_{boe_step}_iter{k}.pdf")
#    plt.show()
    # MAP parameter evaluation
    map_index = np.argmax(trace["log_post"])
    theta_map = trace["theta"][map_index]

