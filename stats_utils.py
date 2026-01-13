import numpy as np
import utils as ut
import scipy
from scipy.stats import chi2, truncnorm, norm, multivariate_normal
from evaluate_models import *
import functools
from itertools import compress
#from qoi_models import pca_var_transformation

#from qoi_models import mass_matrix, A_roi

def calculate_probability(x, prob_dist, log_prob=False):

    x = np.atleast_2d(x)
    M = x.shape[0]
    D = x.shape[1]

    if len(prob_dist) > 1:
        pi_theta = np.zeros((M, D))
        for d, dist in enumerate(prob_dist):
            if log_prob:
                pi_theta[:, d] = dist.logpdf(x[:, d])
            else:
                pi_theta[:, d] = dist.pdf(x[:, d])
        if log_prob:
            pi_theta = pi_theta.sum(axis=1)
        else:
            pi_theta = pi_theta.prod(axis=1)
    elif len(prob_dist) == 1:
        if log_prob:
            pi_theta = prob_dist[0].logpdf(x)
        else:
            pi_theta = prob_dist[0].pdf(x)

#    if pi_theta == 0:
#        pause_here = 1

    return pi_theta



def sqrt_matrix(A):

    evalues, evectors = np.linalg.eig(A)
    assert (evalues >= 0).all()
    sqrt_mat = evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)
    return sqrt_mat


def Mahalanobis_distance(x, mu, sigma):
    """Mahalanobis distance: a measure of the distance
    between point P and distribution D

    Input:
    x - data point
    mu - mean of multivariate normal distribution
    sigma - covariance matrix of multivariate normal distribution

    Output:
    Mahalanobis distance
    """
    x = np.atleast_2d(x)
    mu = np.atleast_2d(mu)

    if x.shape[1] != sigma.shape[0]:
        x = x.T
    assert x.shape[1] == sigma.shape[0]

    if mu.shape[1] != sigma.shape[0]:
        mu = mu.T
    assert mu.shape[1] == sigma.shape[0]

    D = x.shape[1]
    m_dist_x = np.dot((x-mu), np.linalg.inv(sigma))
    m_dist_x = np.dot(m_dist_x, (x-mu).T)

    return m_dist_x



def prob_belongs(x, m_dist_x):
    """ Calculate the probability of a data point belonging
    to a multivariate normal distribution.

    This is calculated as,
    1 - the volume inside the hyper-sphere in Mahalanobis distance
    space defined by the radius given by the Mahalanobis distance of
    the data point from the mean.
    """
    D = len(x)
    prob = 1 - chi2.cdf(m_dist_x, D)

    return prob



def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)



def importance_samples(sampling_dist,\
    nsamples, dim, bounds=None):


    # draw samples from sampling distribution that are
    # within the given bounds
    inputs = None

    if bounds is not None:
        # need to account for some bounds being None and some not
        mn, mx = np.array(bounds).T
    else:
        mn = -np.inf
        mx = np.inf


    if isinstance(sampling_dist, np.ndarray):
        # the sampling distribution is an MCMC chain
        sampling_dist = np.atleast_2d(sampling_dist)
        len_chain = len(sampling_dist)
        thin = len_chain // num_msamples
        samples = sampling_dist[::thin,:]


    else:
        # the sampling distribution is a pdf
        samples = np.zeros((nsamples, dim))
        samples = np.atleast_2d(sampling_dist.rvs(nsamples))
        if samples.shape != (nsamples, dim):
            samples = samples.T

        bad_samples = ((samples < mn) |\
            (samples > mx)).any(1)

        count = 1

        while (bad_samples).any():
            count += 1

            nbad = np.count_nonzero(bad_samples)
            bad_indices = list(compress(range(len(bad_samples)), bad_samples))
            new_samples = \
                np.atleast_2d(sampling_dist.rvs(nbad*100))
            if new_samples.shape != (nbad * 100, dim):
                new_samples = new_samples.T
            good_samples = ((new_samples > mn) &\
                (new_samples < mx)).all(1)
            ngood = np.count_nonzero(good_samples)
            good_indices = list(compress(range(len(good_samples)), good_samples))
            indices = min(ngood, nbad)
            samples[bad_indices[:indices],:] = \
                new_samples[good_indices[:indices]].copy()
            bad_samples = ((samples < mn) |\
                (samples > mx)).any(1)
            if np.mod(count, 2000) == 0:
                print(f"count: {count}")
            if count > 20000:
                raise ValueError("problem with bound and sampling dist")

    return samples



class mvn_dist:

    name = 'mvn'

    def __init__(self, mean, err_pr, **kwargs):

        mean = np.atleast_2d(mean)
        err_pr = np.atleast_2d(err_pr)

        for arg in kwargs.keys():
            if arg == 'ndim':
                ndim = kwargs[arg]
        if 'ndim' not in locals():
            ndim = mean.shape[1]

        self.ndim = ndim

        # collect mean
        if self.ndim > 1:
            if mean.shape[1] == self.ndim:
               self.mean = mean
            if mean.shape[1] == 1:
               # check this works fo all cases
               self.mean = mean * np.ones((1, self.ndim))
            if self.ndim != mean.shape[1]:
                raise valueError('Mean value must be a scalar, integer, ' +\
                   'or have dimension equal to ndim')

        if self.ndim == 1:
            if mean.shape[1] == 1:
                self.mean = mean
            elif mean.shape[1] != 1:
                raise valueError('Mean value must be a scalar, integer, ' +\
                   'or have dimension equal to ndim')

        # collect error precision
        if err_pr.shape[1] == 1:
            if ndim > 1:
                self.err_pr =  err_pr * np.eye(self.ndim)
                if err_pr != 0:
                    self.covar = np.linalg.inv(self.err_pr)
                else:
                    self.covar = self.err_pr
                self.var = np.diag(self.covar)
                self.stdev = self.var**0.5
            else:
                self.err_pr = err_pr
                if err_pr != 0:
                    self.covar = 1/self.err_pr
                else:
                    self.covar = self.err_pr

                self.var = self.covar
                self.stdev = self.var ** 0.5

        elif err_pr.shape[0] == err_pr.shape[1] > 1:
            if ndim > 1:
                self.err_pr = err_pr
                self.covar = np.linalg.inv(self.err_pr)
                self.var = np.diag(self.covar)
                self.stdev = self.var ** 0.5
            else:
                raise valueError('Error precision must be a scalar, integer, ' +\
                   'or have dimension equal to ndim')
        elif err_pr.shape[0] == 1 and err_pr.shape[1] > 1:
            if ndim > 1:
                self.err_pr = np.diag(err_pr.squeeze())
                self.covar = np.linalg.inv(self.err_pr)
                self.var = np.diag(self.covar)
                self.stdev = self.var ** 0.5
            else:
                raise valueError('Error precision must be a scalar, integer, ' +\
                   'or have dimension equal to ndim')


    def pdf(self,x):
        pass


    def logpdf(self, x):
        """ Compute the log likelihood of an N dimensional normal
        distribution at location x

        Parameters:
            - x: ndarray (1 x #dimensions)
            - mean: ndarray (1 x #dimensions)
            - err_pr: error precision, either a scalar
                      or symmetric positive-definite matrix
        """

        def isdiagonal(mat):
            off_diags = np.count_nonzero(mat - np.diag(np.diag(mat)))
            if off_diags == 0:
                return True
            else:
                return False

        x = np.atleast_2d(x)
        mean = np.atleast_2d(self.mean)
        N = self.ndim
        self.err_pr = np.atleast_1d(self.err_pr)

        if isdiagonal(self.err_pr):
            Z = -N * np.log(2*np.pi) + sum(np.log(np.diag(self.err_pr)))
        else:
            Z = -N * np.log(2*np.pi) + np.log(np.linalg.det(self.err_pr))

        self.loglike = 0.5 * (Z - (x - mean) @ self.err_pr @\
            (x - mean).T)

        return self.loglike

    def dlogpdf_dm(self, x):
        """ compute the derivative of the log pdf
            of an n dimensional normal distribution at
            location x wrt mean

        parameters:
            - x: ndarray (1 x #dimensions)
        """

        x = np.atleast_2d(x)
        mean = np.atleast_2d(self.mean)
        n = self.ndim
        self.err_pr = np.atleast_1d(self.err_pr)

        self.dlogpdf_dm = self.err_pr @ (x - mean).T

        return self.dlogpdf_dm

    def dlogpdf_dx(self, x):
        """ compute the derivative of the log pdf
            of an n dimensional normal distribution at
            location x wrt x

        parameters:
            - x: ndarray (1 x #dimensions)
        """

        x = np.atleast_2d(x)
        mean = np.atleast_2d(self.mean)
        n = self.ndim
        self.err_pr = np.atleast_1d(self.err_pr)

        self.dlogpdf_dx = -self.err_pr @ (x - mean).T

        return self.dlogpdf_dx


    def d2logpdf_dm2(self, x):
        """ Compute the second derivative of the log likelihood
            of an N dimensional normal distribution at
            location x wrt physical model

        Parameters:
            - x: ndarray (1 x #dimensions)
        """

        x = np.atleast_2d(x)
        mean = np.atleast_2d(self.mean)
        N = self.ndim
        self.err_pr = np.atleast_1d(self.err_pr)

        self.d2logpdf_dm2 = -self.err_pr

        return self.d2logpdf_dm2


    def d2logpdf_dx2(self, x):
        """ Compute the second derivative of the log likelihood
            of an N dimensional normal distribution at
            location x wrt x

        Parameters:
            - x: ndarray (1 x #dimensions)
        """

        x = np.atleast_2d(x)
        mean = np.atleast_2d(self.mean)
        N = self.ndim
        self.err_pr = np.atleast_1d(self.err_pr)

        self.d2logpdf_dx2 = -self.err_pr

        return self.d2logpdf_dx2



def sample_batch_mvn(mean, cov, size):

    """
    Batch sample multivariate normal distribution.

    Returns: samples from the multivariate normal distributions

    It is not required that "mean" and "cov" have the same shape
    prefix, only that they are broadcastable against each other.

    """

    mean = np.asarray(mean)
    cov = np.asarray(cov)
    size = (size, ) if isinstance(size, int) else tuple(size)
    shape = size + np.broadcast_shapes(mean.shape, cov.shape[:-1])
    X = np.random.standard_normal((*shape, 1))
    L = np.linalg.cholesky(cov)
    return (L @ X).reshape(shape) + mean



def normal_like(mean, err_pr, x, standardize=False):
    """ Compute the likelihood of an N dimensional normal
    distribution at location x

    Parameters:
        - mean: ndarray (1 x N), mean of distribution
        - err_pr: error precision, either a scalar
                  or symmetric positive-definite matrix
        - x:  ndarray (1 x N), where likelihood is being evaluated
    """
    # check if this works for more than one data sample
    def isdiagonal(mat):
        off_diags = np.count_nonzero(mat - np.diag(np.diag(mat)))
        if off_diags == 0:
            return True
        else:
            return False

    x = np.atleast_2d(x)
    mean = np.atleast_2d(mean)
    if np.ndim(x) != np.ndim(mean):
        if np.ndim(x) < np.ndim(mean):
            while np.ndim(x) < np.ndim(mean):
                x = np.expand_dims(x, 0)
        elif np.ndim(mean) < np.ndim(x):
            while np.ndim(mean) < np.ndim(x):
                mean = np.expand_dims(mean, 0)

    assert x.shape[1] == mean.shape[1]
    nsamples = x.shape[0]
    ndist = mean.shape[0]

    N = mean.shape[-1]
    err_pr = np.atleast_1d(err_pr)
    # add check if err_pr is iid
    if np.ndim(err_pr) == 1:
        err_pr_mat = np.diag(np.repeat(err_pr[0], N))
    else:
        assert err_pr.shape[0] == err_pr.shape[1] == N
        err_pr_mat = err_pr

    if isdiagonal(err_pr_mat):
        Z = (2 * np.pi)**(-N/2) * np.prod(np.diag(err_pr_mat))**0.5
    else:
        Z = (2 * np.pi)**(-N/2) * (np.linalg.det(err_pr_mat))**0.5


    if nsamples == 1 and ndist == 1:
        diff = x - mean
        if np.ndim(diff) > 2:
            diff_T = np.moveaxis(diff.T, -1, 0)
        else:
            diff_T = diff.T

        if standardize:
            mat_sqrt = scipy.linalg.sqrtm(err_pr_mat)
            x_tilde = diff @ mat_sqrt
            z_tilde = (2 * np.pi) ** (-N/2)
            likelihood = z_tilde * np.exp(-1/2 * x_tilde @ x_tilde.T)
        else:
            likelihood = Z * np.exp(-1/2 * diff @ err_pr_mat @\
                diff_T)

    elif nsamples > 1 or ndist > 1:
        if np.ndim(mean) < 3:
            mean = np.expand_dims(mean, -1)
        if np.ndim(x) < 3:
            x = np.expand_dims(x, -1)
        diff = x - mean
        if standardize:
            diff_T = np.moveaxis(diff.T, -1, 0)
            mat_sqrt = scipy.linalg.sqrtm(err_pr_mat)
            x_tilde = diff_T @ mat_sqrt
            x_tilde_T = np.moveaxis(x_tilde.T, -1, 0)
            z_tilde = (2 * np.pi) ** (-N/2)
            likelihood = z_tilde * np.exp(-1/2 *  x_tilde @ x_tilde_T)
        else:
            diff_T = np.moveaxis(diff.T, -1, 0)
            likelihood = Z * np.exp(-1/2 * diff_T @ err_pr_mat @\
                diff)

    return likelihood.squeeze()



def normal_loglike(mean, err_pr, x, return_pca=False, standardize=False):
    """ Compute the log likelihood of an N dimensional normal
    distribution at location x

    Parameters:
        - mean: ndarray (1 x N), mean of distribution
        - err_pr: error precision, either a scalar
                  or symmetric positive-definite matrix
        - x:  ndarray (1 x N), where likelihood is being evaluated
    """
    # check if this works for more than one data sample
    def isdiagonal(mat):
        off_diags = np.count_nonzero(mat - np.diag(np.diag(mat)))
        if off_diags == 0:
            return True
        else:
            return False

    x = np.atleast_2d(x)
    mean = np.atleast_2d(mean)
    assert x.shape[1] == mean.shape[1]

    nsamples = x.shape[0]
    ndist = mean.shape[0]

    N = mean.shape[-1]
    err_pr = np.atleast_1d(err_pr)

    # add check if err_pr is iid
    if np.ndim(err_pr) == 1:
        err_pr_mat = np.diag(np.repeat(err_pr, N))
    else:
        assert err_pr.shape[0] == err_pr.shape[1] == N
        err_pr_mat = err_pr

    if isdiagonal(err_pr):
        Z = -N * np.log(2*np.pi) + sum(np.log(np.diag(err_pr_mat)))
    else:
        Z = -N * np.log(2*np.pi) + np.log(np.linalg.det(err_pr_mat))

    if nsamples == 1 and ndist == 1:
        if standardize:
            diff = x - mean
            mat_sqrt = scipy.linalg.sqrtm(err_pr_mat)
            #mat_sqrt = sqrt_matrix(err_pr_mat)
            x_tilde = diff @ mat_sqrt
            z_tilde = -N * np.log(2*np.pi)
            loglike = 0.5 * (z_tilde - x_tilde @ x_tilde.T)
        else:
            loglike = 0.5 * (Z - (x - mean) @ err_pr_mat @\
                (x - mean).T)


    elif nsamples > 1 or ndist > 1:
        if np.ndim(mean) < 3:
            mean = np.expand_dims(mean, -1)
        if np.ndim(x) < 3:
            x = np.expand_dims(x, -1)
            squeeze = True
        diff = x - mean
        if standardize:
            mat_sqrt = scipy.linalg.sqrtm(err_pr_mat)
            diff_T = np.moveaxis(diff.T, -1, 0)
            x_tilde = diff_T @ mat_sqrt
            x_tilde_T = np.moveaxis(x_tilde.T, -1, 0)
            z_tilde = -N * np.log(2*np.pi)
            loglike = 0.5 * (z_tilde - x_tilde @ x_tilde_T)
        else:
            diff_T = np.moveaxis(diff.T, -1, 0)
            loglike = 0.5 * (Z - diff_T @ err_pr_mat @ diff)

    return loglike.squeeze()



def unnorm_log_post(theta, data, err_pr,
    priors, sign='pos', fmodel=None, likelihood='normal',
    nobjectives=1, use_objectives=None, return_pca=False,
    exemplar='mps', standardize=False):


    theta = np.atleast_2d(theta)
    D = theta.shape[1]
    if fmodel is None:
        raise ValueError('A function must be provided for the calculation of the log posterior')
    model_pred = fmodel(theta)

    # calculate log prior
    log_prior = calculate_probability(\
        theta, priors, log_prob=True)

    # calculate the log likelihood
    if exemplar == 'mps':
        log_like = 0
        data[0] = np.atleast_2d(data[0])
#        nload_steps = data[0].shape[0] # for mps make this 1 because data/pred is 1 x (nobs x nstep) x nqoi
        nload_steps = 1
        nqoi = data[0].shape[-1]
        data[0] = data[0].reshape(1, -1, nqoi)
        for step in np.arange(nload_steps):
            for qoi in np.arange(nqoi):
                pred = model_pred[0][step, ..., qoi]
                exp = data[0][step, ..., qoi]
                log_like = normal_loglike(pred,
                    err_pr[0][step, qoi], exp).sum()
    elif exemplar == 'cruciform':
        log_like = 0
        obj_loglike = np.zeros(len(use_objectives))
        nload_steps = data[0].shape[0]
        nqoi = data[0].shape[-1]
        #save_loglike = np.zeros((len(use_objectives), nload_steps, nqoi))
        for idx, ob in enumerate(use_objectives):

            data[idx] = np.atleast_2d(data[idx])
            model_pred[idx] = np.atleast_2d(model_pred[idx])

            for step in np.arange(nload_steps):
                for qoi in np.arange(nqoi):
                    pred = model_pred[idx][step, :, qoi]
                    exp = data[idx][step, :, qoi]
                    pred_qoi = pred
                    exp_qoi = exp
                    #save_loglike[idx, step, qoi] = normal_loglike(pred_qoi, err_pr[idx][step, qoi, ...],
                    #    exp_qoi, return_pca=return_pca, standardize=standardize)
                    obj_loglike[idx] += normal_loglike(pred_qoi, err_pr[idx][step, qoi, ...],
                        exp_qoi, return_pca=return_pca, standardize=standardize)
        log_like = obj_loglike.sum()
    log_post = log_like + log_prior
    if sign == 'neg':
        log_post = log_post * -1.

    if np.isinf(log_post):
        raise ValueError("Log posterior is Inf")

    return log_post



def compute_logpost_gradient(theta, data, noise_var,
        priors, fun_list, sign='pos',
        fmodel=None, likelihood='normal', fd_eps=1e-6, **kwargs):

    """
    compute the gradient of the log posterior with respect to
    parameters theta for function with analytical
    gradient

    The likelihood is assumed to be MVN

    dJ/dtheta = dJl/dm * dm/dtheta + dJp/dtheta
    (1xD) = (1xK)(KxD) + (1xD)
    D = number of dimensions in model
    K = number of models

    Parameters:
    - theta: ndarray of unknown model parameters (pt of gradient calculation location)
    - likelihood: probability distribution
    - prior: 1xD list of probability distributions
    - data: observed data to condition model on:w

    - grad (optional): list of function gradients, if this
      kwarg is passed, then the analytical gradient is
      calculated, otherwise, the FD gradient is calculated

    """

    for arg in kwargs.keys():
        if arg == 'grad' and 'grad' != None:
            grads = kwargs[arg]

    if len(theta.shape) == 1:
        theta = np.atleast_2d(theta)
    data = np.atleast_2d(data)

    D = theta.shape[1]
    N = data.shape[1]
    noise_pr = (1/noise_var) * np.eye(N)

    # dJp_dtheta: derivative of prior wrt parameters theta
    if isinstance(priors, scipy.stats._multivariate.multivariate_normal_frozen):
        prior_means = priors.mean
        prior_var = np.diag(priors.cov)
        par_pr = np.linalg.inv(prior_var)
        dJp_dtheta = par_pr @ (theta - prior_means).T
    else:
        prior_means = np.zeros(D)
        prior_var = np.zeros(D)
        dJp_dtheta = 0
        for p in np.arange(D):
            pr = priors[p]
            prior_mean = pr.mean()
            prior_var = pr.var()
            dJp_dtheta += (theta[:,p] - prior_mean) * (prior_var ** -1)


    # dJl_dm: derivative of likelihood wrt forward model
    qois = np.zeros((1,len(fun_list)))
    for i, fun in enumerate(fun_list):
        qois[0,i] = fun(theta)

# i think fmodel can be removed since a fun_list
# is being passed

#    if fmodel is None:
#        raise ValueError('Must provide function to calculate qois')
#    qois = fmodel(theta)
    if qois.ndim > 2:
        qois = qois.squeeze(axis=0)

    if likelihood == 'normal':
        dJl_dm = (data - qois) @ noise_pr

    # Jacobian of function
    if 'grads' in locals():
        dm_dtheta = ut.compute_jacobian(\
            fun_list,
            theta,
            grad=grads)
    else:
        dm_dtheta = ut.compute_jacobian(\
            fun_list,
            theta,
            fd_eps=fd_eps)

    # dJl_dm is (nsteps x nqoi)
    # dm_dtheta is (nsteps*nqoi) x D
    # reshape dJl_dm so it is 1x(nsteps*nqoi)
    # then product gives (1xD) array to add to
    # (1xD) prior
    dJ_dtheta = dJl_dm.reshape(-1) @ dm_dtheta + dJp_dtheta.T

    # check that norm at gradient is small
    if sign == 'neg':
        dJ_dtheta = dJ_dtheta * -1

    return dJ_dtheta


