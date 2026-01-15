from scipy.stats import bernoulli, norm, uniform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib
import random
import os
import math
from scipy.stats import qmc
import pandas as pd

matplotlib.rcParams.update({'font.size': 22})
# example from andrewcharlesjones.github.io/journal/boed.html
# theta is the threshold for spice tolerance

def rand(start, end, num):
    res = []

    for j in range(num):
        res.append(random.randint(start, end))

    return res

def g_theta(z):
    """ logit function """
    return 1/(1 + np.exp( -z ))

def EIG(theta_nsamples, err_pr, theta_msamples, y_nsamples, x):
    M = theta_msamples.shape[0]
    N = theta_nsamples.shape[0]

    outer_sum = 0
    for nn in range(N):

        # calculate numerator
        data = y_nsamples[nn]
        theta = theta_nsamples[nn]
        m_theta = g_theta(theta - x)
        num = bernoulli(m_theta).pmf(data)

        # calculate denomenator
        theta = theta_msamples
        m_theta = g_theta(theta - x)
        inner_sum = bernoulli(m_theta).pmf(data).sum()

        #inner_sum2 = loglike(data, theta, x).sum()

        outer_sum += np.log(num) - np.log(1/M) - np.log(inner_sum)

    return 1/N * outer_sum

def log_mvn(sample, mean, err_pr):

    N = err_pr.shape[0]
    err_pr_mat = np.diag(err_pr)
    Z = -N * np.log(2*np.pi) + sum(np.log(np.diag(err_pr_mat)))
    log_like = 0.5 * Z - 0.5 * (sample - mean).T * err_pr_mat * \
        (sample - mean)

    return log_like


def log_bern(k, p):
    # since a Bernoulli is a discrete distribution, the likelihood is the pmf
    log_pmf = k * np.log(p) + (1-k) * np.log(1 - p)
    return log_pmf

def loglike(data, theta, x):

    m_theta = g_theta(theta - x) # model evaluated at theta
    log_likelihood = np.log(bernoulli(m_theta).pmf(data))

    return log_likelihood

def log_post(data, theta, mu, sigma, x):
    if 0 < theta:
        m_theta = g_theta(theta - x)
        log_prior = np.log(norm(mu,sigma).pdf(theta))
        log_like = np.log(bernoulli(m_theta).pmf(data)).sum()
        log_post = log_like + log_prior

        if math.isinf(log_post):
            stop_here= 1

    else:
        log_post = -np.inf
    return log_post

def pepper_experiment(x, theta_star=None):
    if theta_star is None:
        theta_star = np.array([20])

    sensitivity = 10
    z = sensitivity * (theta_star - x)
    result = bernoulli(g_theta(z)).rvs()
#    result = np.multiply(np.array(x) <= theta_star, 1)

    return result

def EIG_fig(prior, plot_EIG, x, x_star, step, figdir, posterior=None, theta_star=None ):
    plots = []

    if theta_star is None:
        theta_star = 20

    vals = np.linspace(prior.ppf(.001), prior.ppf(.999), 100)
    fig, ax1 = plt.subplots(figsize=(12,8))
    plt.suptitle(f"Step: {step}")
    ax1.plot(vals, prior.pdf(vals), \
         'b-', lw=2, label="Biasing Dist")
    if posterior is not None:
        ax1.hist(posterior, color="0.5", density=True, label="Posterior")

    ax1.tick_params('y', colors='tab:blue')
    ax2 = ax1.twinx()

    ax2.plot(x[0,:], plot_EIG[0,:], '-ro', label="EIG")
    ax2.plot(x[0,x_star], plot_EIG.max(),
                  'g*', markersize=20, label='Optimal Design')

    ax2.plot([theta_star, theta_star],[0, plot_EIG.max()], 'm-', lw=2, label="True Value")

    ax2.tick_params('y', colors='tab:red')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    ax1.legend(h1+h2, l1+l2, loc="best", fontsize=14)

    fig.tight_layout()
    ax1.set_xlabel("Scoville Score (thousands)")
    ax1.set_ylabel("Prior")
    ax2.set_ylabel("EIG")
    plt.savefig(f"{figdir}/EIG_step_{step}.png")
    #plt.show()
    plt.close("all")

def comp_fig(prior, adaptive_trace, fixed_trace, figdir, nexperiments, title=None):
    vals = np.linspace(prior.ppf(.01), prior.ppf(.99), 100)

    fig, ax1 = plt.subplots(figsize=(12,8))
    ax1.plot(vals, prior.pdf(vals), \
         'r-', lw=3, label="Prior")

    if nexperiments > 1:
        mu = adaptive_trace[:,nexperiments-2].mean()
        sigma = adaptive_trace[:,nexperiments-2].std()
        adaptive_prior = norm(mu, sigma)
        vals2 = np.linspace(adaptive_prior.ppf(.001), adaptive_prior.ppf(.999), 100)
   #     ax1.plot(vals2, adaptive_prior.pdf(vals2), 'b-', lw=3, label="Adaptive Prior")

    n, bins, edges = ax1.hist(adaptive_trace[:,nexperiments-1], color='b', alpha=0.5, density=True,\
        label="Adaptive Design")
    n2, bins2, edges2 = ax1.hist(fixed_trace, color='r', alpha=0.5, density=True, label="Fixed Design")
    ax1.plot([20, 20], [0, n.max()], 'g-', lw=3, label="true value")
    ax1.set_xlabel("Scoville Score (thousands)")
    plt.legend()

    if title is not None:
        fig.suptitle(title)

    plt.savefig(f"{figdir}/DesignComparison_{nexperiments}_Experiments.png")
    plt.close("all")

def model_plot(mu, sigma, figdir):
    # the probability of a pepper being too hot is plotted below, for 5 random
    # samples of theta
    #sample 5 times from the prior
    theta = mu + sigma*np.random.rand(5,1)
    xx = np.arange(1, 35)
    prob = g_theta(theta - xx)

    plt.figure(figsize=(12, 8))
    for curve in range(prob.shape[0]):
        plt.plot(xx, prob[curve,:], marker='o')
    plt.xlabel("Pepper Scoville Score")
    plt.ylabel("Probability of pepper being\n tolerable")
    plt.legend(["Person {}".format(i+1) for i in range(5)])
    plt.savefig(f"{figdir}/Probability of Pepper Being Tolerable.png")
    plt.close("all")

def trace_plots(theta_trace,log_post_trace, step, designType, figdir, title=None):
    _, axes = plt.subplots(1,2, figsize=(12,8), sharey=True)
    axes[1].hist(theta_trace, color="0.5", orientation="horizontal", density=True)

    axes[0].plot(theta_trace)
    axes[0].set_ylabel("theta")
    axes[0].set_xlabel("samples")

    if title is not None:
        axes[0].set_title(title)

    plt.savefig(f"{figdir}/{designType}_Theta_Trace_{step}.png")

    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(log_post_trace)
    ax.set_ylabel("Log Posterior")
    ax.set_xlabel("samples")

    if title is not None:
        fig.suptitle(title)

    plt.savefig(f"{figdir}/{designType}_Log_Posterior_{step}.png")
    plt.close("all")

def plot_stats(fixed_design, adaptive_design, ThetaTrue, figdir):
    nexperiments = fixed_design["Post_mean"].shape[0]
    plt.figure(figsize=(12,8))
    plt.plot(np.arange(nexperiments), fixed_design["Post_mean"], 'r*', markersize=20, label="Fixed")
    plt.plot(np.arange(nexperiments), adaptive_design["Post_mean"], 'bo', markersize=20, label="Adaptive")
    plt.plot([0, nexperiments], [ThetaTrue, ThetaTrue], 'g-', lw=2, label="True Value")
    plt.xlabel("Number of Experiments")
    plt.ylabel("Expected Value of Posterior")
    plt.legend()
    plt.savefig(f"{figdir}/MeanComparison.png")

    plt.figure(figsize=(12,8))
    plt.plot(np.arange(nexperiments), fixed_design["Post_var"], 'r*', markersize=20, label="Fixed")
    plt.plot(np.arange(nexperiments), adaptive_design["Post_var"], 'bo', markersize=20, label="Adaptive")
    plt.xlabel("Number of Experiments")
    plt.ylabel("Variance of Posterior")
    plt.legend()
    plt.savefig(f"{figdir}/VarComparison.png")

    plt.figure(figsize=(12,8))
    plt.plot(np.arange(nexperiments), fixed_design["Post_mean"] - ThetaTrue, 'r*', markersize=20, label="Fixed")
    plt.plot(np.arange(nexperiments), adaptive_design["Post_mean"] - ThetaTrue, 'bo', markersize=20, label="Adaptive")
    plt.xlabel("Number of Experiments")
    plt.ylabel("Bias of Posterior")
    plt.legend()
    plt.savefig(f"{figdir}/BiasComparison.png")
    plt.close("all")

def plot_post_dist(prior, posterior, figdir, title=None):
    nsteps = posterior.shape[1]
    nsamples = posterior.shape[0]

    cmap = plt.get_cmap('winter')
    cNorm = colors.Normalize(vmin=0, vmax=nsteps)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    plt.figure(figsize=(12,8))
    x = np.linspace(0, 45, 1000)

    color = scalarMap.to_rgba(0)
    plt.plot(x, prior.pdf(x), color=color, label="Prior")
    for step in np.arange(nsteps):
        mean = posterior[:,step].mean()
        std = posterior[:,step].std()
        color = scalarMap.to_rgba(step+1)
        y = norm.pdf(x, mean, std)
        if step == nsteps-1:
            plt.plot(x,y, color=color, label="Final Posterior")
        else:
            plt.plot(x, y, color=color)

    plt.plot([20, 20], [0, y.max()], 'm-', alpha=0.75, label="True Value")
    plt.legend()
    plt.xlabel("$\\theta$")
    plt.ylabel("p.d.f.")

    if title is not None:
       plt.title(title)

    print(f"Mean: {mean}")
    print(f"Std:  {std}")
    plt.savefig(f"{figdir}/Posterior_evolution.png")
    plt.close("all")

def MH_MCMC(theta_prior,
            y_current,
            current_design,
            prop_var,
            niter=7000,
            burnin=2000,
            cov_check=1000,
            target_accept=None,
            theta_0=None):

    if target_accept is None:
       target_accept = np.array([0.2, 0.5])

    if theta_0 is None:
        theta_0 = theta_prior.rvs(1)

    trace = {"theta":np.zeros(niter),"log_post":np.zeros(niter)}

    mu = theta_prior.mean()
    sigma = theta_prior.std()

    # now compute the posterior
    curr_post = log_post(y_current, theta_0, mu, sigma, current_design)
    theta = theta_0
    D = theta.shape[0]

    acceptance = np.zeros(niter)
    for iter in range(niter):

        if np.mod(iter, 1000) == 0 :
            print(f"iter = {iter}")

        # covariance adaptation
        if np.mod(iter, cov_check) == 0 and iter > 0:
            accept_rate = (1 + acceptance[iter - cov_check : iter -1].sum())/cov_check
            if accept_rate < target_accept[0] or accept_rate > target_accept[1]:
                prop_var = prop_var * accept_rate / \
                   (target_accept[0] + np.ptp(target_accept[1])/2)

        prop_theta = norm(theta, prop_var ** 0.5).rvs(D)
        prop_post = log_post(y_current, prop_theta, mu, sigma, current_design)

        if np.log(uniform(0,1).rvs(1)) < prop_post - curr_post:
            theta = prop_theta
            curr_post = prop_post
            acceptance[iter] = 1
        else:
            acceptance[iter] = 0

        trace["theta"][iter] = theta
        trace["log_post"][iter] = curr_post

    return trace

# set up fig diri
current_path = os.getcwd()
figdir = os.path.join(current_path, f"hot_pepper_figures")
check_existence = os.path.isdir(figdir)

if not check_existence:
    os.makedirs(figdir)

def CalcEIG(step,
            figdir,
            candidate_designs,
            theta_prior,
            nsamples=10000,
            msamples=100,
            theta_true=None,
            posterior=None):

    mu = theta_prior.mean()
    sigma = theta_prior.std()

    if step == 0:
        """ no data on first step, use prior to compute EIG instead 
        of posterior """

        # compute the EIG for every pepper
        theta_nsamples = theta_prior.rvs(size=nsamples)
        theta_msamples = theta_prior.rvs(size=msamples)

        model_plot(mu, sigma, figdir)

    else:

        # pull samples from posterior of previous step
        l_chain = len(posterior)
        rand_indx1 = rand(0, l_chain-1, nsamples)
        rand_indx2 = rand(0, l_chain-1, msamples)
        theta_nsamples = posterior[rand_indx1]
        theta_msamples = posterior[rand_indx2]

    theta_nsamples = np.expand_dims(theta_nsamples, axis=1)
    theta_msamples = np.expand_dims(theta_msamples, axis=1)

    EIG_store = np.zeros((1,candidate_designs.shape[1]))
    y_nsamples = bernoulli.rvs(g_theta(theta_nsamples - candidate_designs))

    err_pr=None
    for ii in range(candidate_designs.shape[1]):
       EIG_store[0,ii] = EIG(theta_nsamples, err_pr, \
           theta_msamples, y_nsamples[:,ii], candidate_designs[0,ii])

#    norm_EIG = EIG_store/EIG_store.max()
    x_star = np.argmax(EIG_store)
    opt_design = candidate_designs[0, x_star]

    return x_star, opt_design, EIG_store

# we wish to run only 10 rounds of the experiment, where we select 1 pepper each round
# we are allowed to select the same pepper on multiple rounds

theta_true = np.array([20])
design_min = 6
design_max = 36
candidate_designs = np.arange(design_min,design_max)[np.newaxis] # 20 peppers with evenly spaced Scoville scores

# define prior for theta - keyword loc = mean, scale = stdev
mu = 15
sigma = 8
theta_prior = norm(mu, sigma)

# MCMC settings
niter = 100
burnin = 1
cov_check = 1000
target_accept = np.array([0.2, 0.5])

#EIG Settings
M = 100
N = 100
current_design = []
y_current = []

# BOED settings
nsteps = 5

# initialize dictionaries for traces and stats
trace = {"theta":np.zeros((niter,nsteps)),"log_post":np.zeros((niter,nsteps))}

adaptive_stats = {"Post_var": np.zeros(nsteps), "Post_mean": np.zeros(nsteps), \
    "Design": np.zeros(nsteps)}

for step in range(0,nsteps):
    print("********************")
    print(f"step: {step + 1}")

    if step == 0:
        post = None
    else:
        post = trace["theta"][burnin:,step-1]

    x_star, opt_design, EIG_store  = CalcEIG(step,
           figdir,
           candidate_designs,
           theta_prior,
           nsamples=N,
           msamples=M,
           theta_true=theta_true,
           posterior=post )


    if step == 0:
        EIG_fig(theta_prior, EIG_store, candidate_designs, x_star, step, figdir, posterior=post)
    else:
        dist = norm(mu, sigma)
        EIG_fig(dist, EIG_store, candidate_designs, x_star, step, figdir, posterior=post)
    #now perform experiment with x_star
    current_design.append( opt_design )
    y_current.append( pepper_experiment( opt_design, theta_star=theta_true)  )
    print(f"Current Design: {current_design}")

    # now compute the posterior
    theta_0 = theta_prior.rvs(1)
    prop_var = 10
    # call MH_MCMC
    new_trace = MH_MCMC(theta_prior,\
                        y_current,\
                        current_design,\
                        prop_var,\
                        niter=niter,\
                        burnin=burnin,\
                        cov_check=cov_check,\
                        theta_0=theta_0)

    trace["theta"][:,step] = new_trace["theta"]
    trace["log_post"][:,step] = new_trace["log_post"]

    # update prior - assume normal distribution
    mu = trace["theta"][burnin:,step].mean()
    sigma = trace["theta"][burnin:,step].std()

    # update prior
#    theta_prior = norm(mu, sigma)

    adaptive_stats["Post_var"][step] = sigma ** 2
    adaptive_stats["Post_mean"][step] = mu
    adaptive_stats["Design"][step] = current_design[-1]
    label = f"Adaptive: {step + 1} Experiments Performed"
    trace_plots(trace["theta"][:,step], trace["log_post"][:,step], step, 'Adaptive', figdir, title=label)

label = f"Adaptive Design: Prior to Posterior in {nsteps} Steps"
plot_post_dist( norm(15,8), trace["theta"], figdir, title=label )


# Fixed design
mu = 15
sigma = 8
theta_prior = norm(mu, sigma)
l_bound = 0
u_bound = design_max
sampler = qmc.Halton(d=1, scramble=False)
current_design1 = []
y_current1 = []
ndesigns = nsteps

fixed_stats = {"Post_var": np.zeros(ndesigns), "Post_mean": np.zeros(ndesigns), \
    "Designs": np.zeros(ndesigns)}

for index in range(ndesigns):
    print("***************")
    print(f"Fixed Design: step {index + 1}")
    design_sample = qmc.scale( sampler.random(n=1), l_bound, u_bound )
    design_sample = round(design_sample[0][0])
    current_design1.append( design_sample )
    # = candidate_designs[0,::fixed_designs[index]]
    print(f"Current Design: {current_design1}")
    y_current1.append(  pepper_experiment( design_sample, theta_star=theta_true ) )
    theta_0 = theta_prior.rvs(1)
    prop_var = 10

    fixed_trace = MH_MCMC(theta_prior,
                          y_current1,
                          current_design1,
                          prop_var,
                          niter=niter,
                          burnin=burnin,
                          cov_check=cov_check,
                          theta_0=theta_0)

    fixed_stats["Post_var"][index] = fixed_trace["theta"][burnin:].var()
    fixed_stats["Post_mean"][index] = fixed_trace["theta"][burnin:].mean()    
    fixed_stats["Designs"][index] = design_sample

    nexperiments = len(current_design1)
    label = f"Fixed: {nexperiments} Experiments Performed"
    trace_plots(fixed_trace["theta"], fixed_trace["log_post"], nexperiments, 'Fixed', figdir, title=label)
    label = f"Fixed vs Adaptive: {nexperiments} Experiments Performed"
    comp_fig(theta_prior, trace["theta"][burnin:,:], fixed_trace["theta"][burnin:], figdir,\
       nexperiments, title=label)

plot_stats(fixed_stats, adaptive_stats, theta_true, figdir)
adaptive_df = pd.DataFrame.from_dict(adaptive_stats)
fixed_df = pd.DataFrame.from_dict(fixed_stats)

