import numpy as np
from scipy.stats import multivariate_normal, norm
from posterior_update_tools import Laplace_approx
from functools import partial
from math import isinf
from stats_utils import normal_loglike, calculate_probability, importance_samples, normal_like, sample_batch_mvn
from utils import isPD, nearestPD
import warnings
import qoi_models
import time
from qoi_models import displacement_integration
import pickle
import sys
import matplotlib
import matplotlib.pyplot as plt
import copy
import random
#np.seterr(all='warn')
#warnings.simplefilter('error')



def unpack_nested_mc_settings(nested_mc_settings):

    M = nested_mc_settings['M']
    N = nested_mc_settings['N']
    ind_inner_samples = nested_mc_settings['ind_inner_samples']
    ind_inner_outer_samples = nested_mc_settings['ind_inner_outer_samples']
    evidence_method = nested_mc_settings['evidence_method']
    biasing_dist = nested_mc_settings['biasing_dist']

    if evidence_method != 'importance_sampling' and evidence_method != 'Laplace':
        raise ValueError('evidence method not implemented')

    if evidence_method == 'importance_sampling' and biasing_dist is None:
        raise ValueError('must provide biasing distribution for importance sampling')

    if not ind_inner_outer_samples and ind_inner_samples:
        print('Inner samples defaulting to non-independent to agree with non-independent inner-outer samples')
        ind_inner_samples = False

    if not ind_inner_outer_samples and M != N:
        print('M defaulting to N to agree with non-independend inner-outer samples')
        M = N

    if biasing_dist == 'self_normalizing' or biasing_dist == 'laplace_based'\
        or biasing_dist == 'self_centered':
        ind_inner_samples = True
        ind_inner_outer_samples = True

    return N, M, ind_inner_samples, ind_inner_outer_samples,\
        evidence_method, biasing_dist



def calculate_ess(weights, normalized=True):

    if normalized:
        ess = 1 / np.square(weights).sum()

    else:
        ess = np.square(weights.sum()) / np.square(weights).sum()

    return ess



def self_centered_biasing_dist(i, parameter_samples):

    u_i_post = parameter_samples[i]

    cov_i_post = np.cov(parameter_samples.T)

    if not isPD(cov_i_post):
        cov_bk = cov_i_post.copy()
        cov_i_post = nearestPD(cov_i_post)
        cov_mape = np.divide(np.abs(cov_i_post - cov_bk), np.abs(cov_bk)).mean()*100

    try:
        q_x = multivariate_normal(u_i_post, cov_i_post)

    except:
        if isPD(cov_i_post):
            q_x = multivariate_normal(u_i_post, cov_i_post,
                allow_singular=True)

    return q_x



def self_normalizing_biasing_dist(i, parameter_samples,
    model_qois, noise_var, num_qois=[1], nobjectives=1,
    use_objectives=[0], EIG_prior_dist=None, des=None):

    N = parameter_samples.shape[0]
    D = parameter_samples.shape[1]

    # calculate estimated posterior mean: \mu_hat_{post}^{i}
    # the denomenator is the same for each k=1,...,N
    likelihood_y_i_given_thetas = np.ones(N)
    for idx, ob in enumerate(use_objectives):
        nv = np.atleast_1d(noise_var[ob])
        if np.ndim(nv) == 1:
            err_pr = np.full((1, num_qois[ob]), 1/nv)
        elif np.ndim(nv) > 1:
            if np.ndim(nv) == 2:
                nv = np.expand_dims(nv, (0, 1))
            if np.ndim(nv) == 3:
                nv = np.expand_dims(nv, 0)
            err_pr = np.linalg.inv(nv)
        for qoi in np.arange(num_qois[ob]):
            mean = model_qois[idx][..., qoi]
            x = np.atleast_1d(model_qois[idx][i, ..., qoi])
            x = np.expand_dims(x, 0)
            likelihood_y_i_given_thetas *= normal_like(mean, err_pr[0, qoi, ...], x)

    norm_factor = 1/N * likelihood_y_i_given_thetas.sum()
#    norm_factor = likelihood_y_i_given_thetas.sum()

    # calcualte w_k, k = 1,...,N
    w_k = likelihood_y_i_given_thetas / norm_factor
#    w_k_squared = w_k ** 2
#    ess = N**2 * 1/w_k_squared.sum()
#    ess = 1/w_k_squared.sum()
#    cutoff = 1 + 50/N
#    if ess < cutoff:
#        print('reverting to self-centered biasing distribution due to low effective sample size')
#        q_evd = self_centered_biasing_dist(i, parameter_samples)
#        return [q_evd]


    u_k = parameter_samples * w_k[:, np.newaxis]
    u_i_post = 1/N * u_k.sum(axis=0)
    #u_i_post = u_k.sum(axis=0)

    cov_i_post = 1/N * (parameter_samples - u_i_post).T @\
        np.eye(N)*w_k @ (parameter_samples - u_i_post)
#    cov_i_post =  (parameter_samples - u_i_post).T @\
#        np.eye(N)*w_k @ (parameter_samples - u_i_post)

    if not isPD(cov_i_post):
#        print(f'nn = {i}, des = {des}: nonPSD cov-mat --> reverting to self-centered biasing distribution')
        q_evd = self_centered_biasing_dist(i, parameter_samples)
        return [q_evd]

    try:
        q_x = multivariate_normal(u_i_post, cov_i_post)

    except:
        q_x = multivariate_normal(u_i_post, cov_i_post,
            allow_singular=True)

    return [q_x]


def laplace_based_biasing_dist(
    data, fmodel, noise_var, design,
    priors, theta_0=None, bounds=None,
    fprime=None, fd_eps=1e-6,
    likelihood='normal', use_own_fd=True, factr=1e7,
    boe_step=None, nrestarts=[1,3], nobjectives=1,
    use_objectives=[0], fd_order=4, return_pca=False,
    num_qois=1, exemplar='mps'):

    q_x, theta_map, cov, log_evidence =\
        Laplace_approx(data, fmodel,
        noise_var, priors, bounds=bounds, fprime=None,
        fd_eps=fd_eps, likelihood='normal',
        use_own_fd=use_own_fd, factr=1e7, boe_step=None,
        nrestarts=nrestarts, nobjectives=nobjectives,
        use_objectives=use_objectives, fd_order=fd_order,
        return_pca=return_pca, num_qois=num_qois,
        design=design, exemplar=exemplar)

    return [q_x]



def calculate_EIG_num(N, nqois, y_nsamples, nobjectives,
    use_objectives, design, noise_var, num_qoi, return_pca=False,
    standardize=False):

#    if nobjectives == 1:
#        nobs = [nqois[0].shape[1]]
#    else:
#        nobs = [nqois[0].shape[1], nqois[1].shape[1]]
    nobs = [nqois[idx].shape[1] for idx in np.arange(len(use_objectives))]

    if return_pca and 0 in use_objectives:
        disp_noise_var = qoi_models.pca_noise_var(design,
            noise_var[0], num_qoi[0], return_steps='last')
        noise_var = [disp_noise_var, noise_var[1]]
#    ndim = len(nqois[0].reshape(-1)) # num dimension in data

    # calculate the numerator
    num = np.zeros(N)
    for idx, ob in enumerate(use_objectives):
        nv = np.atleast_1d(noise_var[ob])
        if np.ndim(nv) == 1:
            err_pr = np.full((1, num_qoi[ob]), 1/nv)
        elif np.ndim(nv) > 1:
            err_pr = np.linalg.inv(nv)
        for qoi in np.arange(num_qoi[ob]):
            mean = nqois[idx][..., qoi]
            x = y_nsamples[idx][..., qoi]
            num += normal_loglike(mean,
                err_pr[0, qoi, ...], x, standardize=standardize)

    return num



def calculate_EIG_denom(outer_iter, M, mqois, data_sample, evidence_weights,
    nobjectives, use_objectives, candidate_design, noise_var, num_qoi,
    fmodel, sequential_design,
    evidence_method='importance_sampling', return_pca=False,
    verbose=False, boe_step=None, fmodel_des_fixed=None,
    priors=None, bounds=None, fprime=None, fd_eps='auto',
    use_own_fd=True, likelihood='normal', fd_order=2,
    nrestarts=[1,1], underflow_handling=False,
    standardize=False, exemplar='mps'):


    if evidence_method == 'importance_sampling':

        inner_like = np.ones(M)
        inner_loglike = np.zeros(M)
        for idx, ob in enumerate(use_objectives):
            nv = np.atleast_1d(noise_var[ob])
            if np.ndim(nv) == 1:
                err_pr = np.full((1, num_qoi[ob]), 1/nv)
            elif np.ndim(nv) > 1:
                if np.ndim(nv) == 2:
                    nv = np.expand_dims(nv, (0, 1))
                err_pr = np.linalg.inv(nv)
            for qoi in np.arange(num_qoi[ob]):
                mean = mqois[idx][..., qoi]
                x = data_sample[idx][..., qoi]
                #inner_like *= normal_like(mean, err_pr[0, qoi, ...], x, standardize=standardize)
                inner_loglike += normal_loglike(mean, err_pr[0, qoi, ...], x, standardize=standardize)

        #inner_like = inner_like * evidence_weights
        inner_loglike = inner_loglike + np.log(evidence_weights)
        max_val = inner_loglike.max()
        inner_sum2 = np.exp(inner_loglike - max_val).sum()

        #inner_sum = inner_like.sum()

        if np.isinf(inner_sum2) or inner_sum2 == 0:
            if underflow_handling:
                print(f"Underflow nn = {outer_iter}: inner sum set to 10**-323.")
                inner_sum2 = 10**-323
            else:
                return 0
        #denom = np.log(inner_sum)
        denom = max_val + np.log(inner_sum2)

    # Laplace approximation for evidence
    elif evidence_method == 'Laplace':
        if sequential_design:
            return_steps = "last"
        else:
            return_steps = "all"

        post, theta_map, cov, log_evidence = \
            Laplace_approx(data_sample, fmodel_des_fixed,
            noise_var, priors,
            bounds=bounds, fprime=fprime,
            fd_eps=fd_eps, use_own_fd=use_own_fd,
            likelihood=likelihood, fd_order=fd_order,
            nobjectives=nobjectives, use_objectives=use_objectives,
            nrestarts=nrestarts, return_pca=return_pca,
            design=candidate_design, num_qois=num_qoi,
            return_steps=return_steps,
            exemplar=exemplar)

        denom = log_evidence

    else:
        raise ValueError("Evidence method not implemented.")

    return denom



def get_candidate_design(current_design,
    next_design, boe_step, sequential_design=True,
    continuous_design=True):

    if continuous_design:
        cand_des = current_design.copy()
        cand_des.append(next_design)
        ndes_steps = boe_step+1

        if sequential_design:
            return_steps = 'last'
        else:
            return_steps = 'all'
    else:
        if sequential_design:
            # sample from current design
            cand_des = next_design
            ndes_steps = 1
            return_steps = 'last'
        else:
            # sample from full design
            cand_des = current_design.copy()
            cand_des.append(next_design)
            return_steps = 'all'

    return cand_des, return_steps



def draw_samples(sampling_dist, D, nsamples, sampling_bounds, posterior_method='Laplace'):
    try:
        if posterior_method == 'Laplace':
            if len(sampling_dist) > 1:
                samples = np.zeros((nsamples, D))
                for d, dist in enumerate(sampling_dist):
                    samples[:, d] = \
                        importance_samples(\
                            dist, nsamples, 1, bounds=sampling_bounds[d]).squeeze()
            elif len(sampling_dist) == 1:
                samples = importance_samples(\
                sampling_dist[0], nsamples, D, bounds=sampling_bounds)
        elif posterior_method == 'MCMC':
            length_chain = len(sampling_dist[0])
            random_chain_indices = random.sample(range(1, length_chain), nsamples)
            samples = sampling_dist[0][random_chain_indices]
    except:
        samples = None
    return samples



def get_inner_loop_qois(q_evd, design_options, D, M, bounds,
    EIG_prior_dist, use_objectives, nobjectives,
    fmodel_des_fixed, exemplar='mps', posterior_method='Laplace'):


    nbiasing_dists = len(q_evd)
    ndesign_options = len(design_options)
    w_hat = {}
    msample_qois = {}

    theta_msamples = [None] * ndesign_options
    # draw inner samples and calculate qois
    if nbiasing_dists == 1:
        # same inner samples used for all design options
        # (prior and self-centered biasing dists)
            bias_dist = q_evd[0]
            inner_samples = draw_samples(bias_dist, D, M, bounds, posterior_method=posterior_method)
            if inner_samples is None:
                return [None, None]

            pi_theta = calculate_probability(\
                inner_samples, EIG_prior_dist)
            q_theta = calculate_probability(\
                inner_samples, bias_dist)
            if not q_theta.all():
                raise ValueError("zero in q_theta")
            w_tilde = np.divide(pi_theta, q_theta)
            norm_constant = M

            for d, des in enumerate(design_options):
                theta_msamples[d] = inner_samples
                w_hat[des] = w_tilde / norm_constant
#                                ess = calculate_ess(w_hat[des])

    elif nbiasing_dists > 1:
        # unique inner samples used for each design option
        # (self-normalized and laplace-based biasing dists)
        assert nbiasing_dists == len(design_options)
        for d, bias_dist in enumerate(q_evd):
            theta_msamples[d] = draw_samples(\
                bias_dist, D, M, bounds, posterior_method=posterior_method)
            if theta_msamples[d] is None:
                return [None, None]

            pi_theta = calculate_probability(\
                theta_msamples[d], EIG_prior_dist)
            q_theta = calculate_probability(\
                theta_msamples[d], bias_dist)

            if not q_theta.all():
                raise ValueError("zero in q_theta")

            w_tilde = np.divide(pi_theta, q_theta)
            norm_constant = M
            w_hat[design_options[d]] = w_tilde / norm_constant
#                            ess = calculate_ess(w_hat[design_options[d]], normalized=False)

    for d, des in enumerate(design_options):
        msample_qois[des] = [None] * len(use_objectives)
        if exemplar == 'mps':
            msample_qois[des][0] = fmodel_des_fixed[d](\
                theta_msamples[d].reshape(-1, D))[0]
        elif exemplar == 'cruciform':
            msample_qois[des] = fmodel_des_fixed[d](theta_msamples[d])
            for idx, ob in enumerate(use_objectives):
                msample_qois[des][idx] = msample_qois[des][idx][0, ...]

    return msample_qois, w_hat



def EIG(current_design, design_options, D, boe_step,
    EIG_prior_dist, theta_priors,
    fmodel, fd_eps='auto', use_own_fd=True, fprime=None,
    bounds=None, posterior_method='Laplace',
    sequential_design=True, likelihood='normal', noise_var=None,
    continuous_design=True, num_qois=[1], nested_mc_settings=None,
    nobjectives=1, use_objectives=[0], theta_true=None, fd_order=2,
    return_pca=False, verbose=True, theta_0=None,
    nrestarts=[1,1], return_log_evidence=False, q_evd=None,
    surrogate_type='gps', surrogate_dir=None,
    exemplar='mps', standardize=False,
    integrate_displacement=False):

    # EIG calculated with PCA modes because it is much faster than the field space
    if not integrate_displacement:
        return_pca = True

    # extract nested MC settings
    if nested_mc_settings is None:
        raise ValueError('Must provide nested MC settings.')

    N, M, ind_inner_samples, ind_inner_outer_samples,\
        evidence_method, biasing_dist = \
            unpack_nested_mc_settings(nested_mc_settings)


    # draw parameter samples for outer loop (same samples for all designs)
    theta_nsamples = draw_samples(EIG_prior_dist, D, N, bounds, posterior_method=posterior_method)
    if theta_nsamples.shape[0] != N:
        theta_nsamples = theta_nsamples.T
    assert len(theta_nsamples) == N

    if False:
        # for checking evidence
        theta_nsamples[0] = theta_true

    EIG_num = {}
    EIG_denom = {}
    EIG = [None] * len(design_options)
    ess = {}
    nsample_qois = {}
    y_nsamples = {}

    noise_var_bk = noise_var.copy()
    cand_des = [None] * len(design_options)
    fmodel_des_fixed = [None] * len(design_options)
    noise_var_transformed = [None] * len(design_options)
    EIG_denom_noise_var = [None] * len(design_options)

    for d, des in enumerate(design_options):

        ess[des] = np.zeros((N))
        EIG_denom[des] = np.zeros((N))

        cand_des[d], return_steps = get_candidate_design(\
            current_design, des, boe_step,
            sequential_design=sequential_design,
            continuous_design=continuous_design)

        # fmodel produces N x (nobs_pstep * nboe_steps) x nqoi
        # calculate qois for outer loop samples
        if exemplar == 'mps':
            nsample_qois[des] = fmodel(\
                theta_nsamples, cand_des[d], return_steps=return_steps)
            fmodel_des_fixed[d] = partial(fmodel,
                design=cand_des[d], return_steps=return_steps)
        elif exemplar == 'cruciform':
            surr= qoi_models.load_cruciform_surrogate(cand_des[d],
                surrogate_type=surrogate_type, return_steps=return_steps,
                surrogate_dir=surrogate_dir)
            fmodel_des_fixed[d] = partial(fmodel, surr=surr, return_pca=return_pca)

            # return field data so noise can be added in the field space
            nsample_qois[des] = fmodel(\
                theta_nsamples, surr,
                integrate_displacement=False,
                return_pca=False)

            for idx, ob in enumerate(use_objectives):
                nsample_qois[des][idx] = nsample_qois[des][idx][0, ...]

        # draw samples from data distribution (adding noise)
        if likelihood == 'normal':
            y_nsamples[des] = [None] * len(use_objectives)
            for idx, ob in enumerate(use_objectives):
                # nsamples x nobs x nqoi
                y_nsamples[des][idx] = norm(nsample_qois[des][idx],
                    noise_var[ob]**0.5).rvs()
        else:
            raise ValueError('likelihood distriubtion not implemented')

        # perform tranformations/dimension reduction
        if any(key.startswith('integrate_displacement') for key in fmodel.keywords):
            if fmodel.keywords['integrate_displacement']:
                y_nsamples[des][0] =\
                    displacement_integration(y_nsamples[des][0])
                nsample_qois[des][0] =\
                    displacement_integration(nsample_qois[des][0])

        # dimension reduction
        if return_pca and 0 in use_objectives:
            y_nsamples[des][0] =\
                qoi_models.experimental_data_2_pca(\
                    np.expand_dims(y_nsamples[des][0], 0), cand_des[d]).squeeze()
            nsample_qois[des][0] =\
                 qoi_models.experimental_data_2_pca(\
                     np.expand_dims(nsample_qois[des][0], 0), cand_des[d]).squeeze()

        # calculate numerator of the EIG MC approximation
        EIG_num[des] = calculate_EIG_num(N, nsample_qois[des],
            y_nsamples[des], nobjectives, use_objectives, cand_des[d],
            noise_var, num_qois, return_pca=return_pca, standardize=standardize)

        nobs = [None] * len(use_objectives)
        for idx, ob in enumerate(use_objectives):
            nobs[idx] = nsample_qois[des][idx].shape[1]

        if return_pca and 0 in use_objectives:
            disp_noise_var = qoi_models.pca_noise_var(cand_des[d],
                noise_var[0], num_qois[0], return_steps='last')
            noise_var_transformed[d] = [disp_noise_var, noise_var[1]]
        else:
            noise_var_transformed[d] = noise_var
        # end design loop for nsample_qois and y_nsamples

    # loop through outer samples, and draw inner samples
    # calculate EIG denominator
    for d in np.arange(len(design_options)):
        if evidence_method == 'importance_sampling':
            EIG_denom_noise_var[d] = noise_var_transformed[d]
        elif evidence_method == 'Laplace':
            EIG_denom_noise_var[d] = noise_var #transformation of noise happens in Laplace_approx()

    inner_loop_start = time.time()
    print("Starting inner loop EIG calculations...")
    for nn in np.arange(0, N):

        biasing_dist_nn = biasing_dist
        underflow_handling = False

        if False:
            inc = 100
            if np.mod(nn, inc) == 0 and nn > 0:
                inner_loop_end = time.time()
                inner_loop_time = inner_loop_end - inner_loop_start
                print(f"nn = {nn},  time to calculate last {inc} inner_loop EIG: {inner_loop_time}")
                inner_loop_start = time.time()

        while True:
            # draw inner samples - repeat with new biasing distribution if underflow occurs in EIG denominator
            if evidence_method != 'importance_sampling':
                w_hat = {}
                msample_qois = {}
                for idx, des in enumerate(design_options):
                    msample_qois[des] = None
                    w_hat[des] = None
                    ess[des] = None
            elif evidence_method == 'importance_sampling':
                if biasing_dist_nn == 'prior':
                    if ind_inner_outer_samples: # then draw samples for inner loop
                        if ind_inner_samples or nn == 0:
                            q_evd = [EIG_prior_dist]
                            msample_qois, w_hat = get_inner_loop_qois(q_evd, design_options,
                                D, M, bounds, EIG_prior_dist, use_objectives,
                                nobjectives, fmodel_des_fixed, exemplar=exemplar, posterior_method=posterior_method)
                            msample_qois_bk = msample_qois.copy()
                            w_hat_bk = w_hat.copy()
                        else:
                            # CONFIRM THIS LINE OF CODE
                            msample_qois = msample_qois_bk.copy()
                            w_hat = w_hat_bk.copy()

                    elif not ind_inner_outer_samples and nn == 0:
                        assert M == N
                        # M and N samples are the same, therefore model output is the same
                        msample_qois = {}
                        w_hat = {}
                        for d, des in enumerate(design_options):
                            msample_qois[des] = [None] * len(use_objectives)
                            for idx, ob in enumerate(use_objectives):
                                msample_qois[des][idx] = nsample_qois[des][idx].copy()
                            w_tilde = np.ones(M)
                            norm_constant = M #w_tilde.sum()
                            w_hat[des] = w_tilde / norm_constant


                elif biasing_dist_nn == 'self_centered':
                    q_evd = [[self_centered_biasing_dist(nn,
                        theta_nsamples)]]
                    msample_qois, w_hat = get_inner_loop_qois(q_evd, design_options,
                        D, M, bounds, EIG_prior_dist, use_objectives,
                        nobjectives, fmodel_des_fixed, exemplar=exemplar, posterior_method=posterior_method)

                elif biasing_dist_nn == 'laplace_based':
                    q_evd = [None] * len(design_options)
                    for d, des in enumerate(design_options):
                        data_nn = [np.expand_dims(y_nsamples[des][ob][nn], axis=0) for ob in np.arange(len(use_objectives))]
                        q_evd[d] = laplace_based_biasing_dist(\
                            data_nn,
                            fmodel_des_fixed[d],
                            noise_var,
                            cand_des[d],
                            EIG_prior_dist,
                            bounds=bounds,
                            fprime=fprime,
                            fd_eps=fd_eps,
                            likelihood='normal',
                            use_own_fd=use_own_fd,
                            factr=1e7,
                            boe_step=boe_step,
                            nobjectives=nobjectives,
                            use_objectives=use_objectives,
                            fd_order=fd_order,
                            nrestarts=[1,1],
                            return_pca=return_pca,
                            num_qois=num_qois,
                            exemplar=exemplar)
                    msample_qois, w_hat = get_inner_loop_qois(q_evd, design_options,
                        D, M, bounds, EIG_prior_dist, use_objectives,
                        nobjectives, fmodel_des_fixed, exemplar=exemplar, posterior_method=posterior_method)


                elif biasing_dist_nn == 'self_normalizing':
                    q_evd = [None] * len(design_options)
                    for d, des in enumerate(design_options):
                        q_evd[d] =\
                            self_normalizing_biasing_dist(\
                            nn, theta_nsamples,
                            y_nsamples[des],
                            noise_var_transformed[d],
                            num_qois=num_qois,
                            nobjectives=nobjectives,
                            use_objectives=use_objectives,
                            des=des)
                    msample_qois, w_hat = get_inner_loop_qois(q_evd, design_options,
                        D, M, bounds, EIG_prior_dist, use_objectives,
                        nobjectives, fmodel_des_fixed, exemplar=exemplar, posterior_method=posterior_method)
                # end loop for drawing inner samples and calcualting qois

            # calcualte EIG denominator
            underflow = False
            sampling_fail = False

            for d, des in enumerate(design_options):

                if msample_qois is None:
                    sampling_fail = True
                    break

                y_n = [np.atleast_1d(np.expand_dims(\
                    y_nsamples[des][idx][nn, ...], axis=0)) for idx in np.arange(len(use_objectives))]
                EIG_denom[des][nn] = calculate_EIG_denom(nn, M, msample_qois[des],
                    y_n, w_hat[des], nobjectives, use_objectives, cand_des[d],
                    EIG_denom_noise_var[d], num_qois, fmodel,
                    sequential_design, evidence_method=evidence_method,
                    return_pca=return_pca, boe_step=boe_step,
                    fmodel_des_fixed=fmodel_des_fixed[d],
                    priors=EIG_prior_dist,
                    bounds=bounds, underflow_handling=underflow_handling,
                    standardize=standardize, exemplar=exemplar)

                if EIG_denom[des][nn] == 0:
                    underflow = True
                    underflow_des = des
                    break


            dist_order = ['prior', 'laplace_based', 'self_centered', 'self_normalizing']
            if underflow or sampling_fail:
                if biasing_dist_nn == dist_order[0]:
                    biasing_dist_nn = dist_order[1]
                    if verbose:
                        if underflow:
                            print(f"Underflow at nn = {nn}, des {underflow_des}, biasing dist changed from {dist_order[0]} to {dist_order[1]}")
                        elif sampling_fail:
                            print(f"Problem drawing inner samples at nn = {nn},  biasing dist changed from {dist_order[0]} to {dist_order[1]}")
                    continue
                elif biasing_dist_nn == dist_order[1]:
                    biasing_dist_nn = dist_order[2]
                    if verbose:
                        if underflow:
                            print(f"Underflow at nn = {nn}, des {underflow_des}, biasing dist changed from {dist_order[1]} to {dist_order[2]}")
                        elif sampling_fail:
                            print(f"Problem drawing inner samples at nn = {nn},  biasing dist changed from {dist_order[1]} to {dist_order[2]}")
                    continue
                elif biasing_dist_nn == dist_order[2]:
                    biasing_dist_nn = dist_order[3]
                    if verbose:
                        if underflow:
                            print(f"Underflow at nn = {nn}, des {underflow_des} biasing dist changed from {dist_order[2]} to {dist_order[3]}")
                        elif sampling_fail:
                            print(f"Problem drawing inner samples at nn = {nn}, biasing dist changed from {dist_order[2]} ot {dist_order[3]}")
                    continue
                else:
                    if underflow:
                        underflow_handling = True
                        if verbose:
                            print(f"Underflow override turned on")
                        continue
                    elif sampling_fail:
                        EIG_denom[des][nn] = 0
                        EIG_num[des][nn] = 0
            else:
                break

                # for checking log evidence - remove
#                if verbose:
#                    if nn == 0 and des == 10.0:
#                        print(f"IS log evidence: {EIG_denom[des][0]}")

    for d, des in enumerate(design_options):
        EIG_i = EIG_num[des] - EIG_denom[des]
        outer_sum = EIG_i.sum()
        EIG[d] = 1/N * outer_sum

    if return_log_evidence:
        return EIG, EIG_denom[des][0]
    else:
        return EIG

