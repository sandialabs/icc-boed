import numpy as np
from numpy import linalg as la
from scipy.spatial.distance import cdist
from scipy.linalg import cho_solve
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib
from scipy.stats._multivariate import _eigvalsh_to_eps
import scipy
import copy
import pickle

def nearestPD(A):
    """
    Find the nearest positive-definite matrix to input

    https://stackoverflow/com/questions/43238173/python-convert-matrix-to-positive-semi-definite

    A Python/Numpy port of John D'Errico's 'nearestSPD' MATLAB
    code [1] which credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite matrix" (1988):
    https: //doi.org/10.1016/0024-3795(88)90223-6
    """

    # symmetrize A into B
    B = (A + A.T) / 2

    # Compute the symmetric polar factor of B. Call it H.
    # H is itself PSD
    _, s, V = la.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))

    # Get Ahat in the above forumal
    Ahat = (B + H) / 2

    # ensure symmetry
    A2 = (Ahat + Ahat.T) / 2

    # test that A2 is PD. If it is not, then tweak
    # it just a bit.
#    if isPD(A2):
    if isPD(A2, methods=['scipy', 'eigvals']):
        return A2

    spacing = np.spacing(la.norm(A))

    I = np.eye(A.shape[0])
    k = 1
#    while not isPD(A2):
    while not isPD(A2, methods=['scipy', 'eigvals']):
        mineig = np.min(np.real(la.eigvals(A2)))
        A2 += I * (-mineig * k**2 + spacing)
        k += 1


    return A2


def isPD(A, methods=['all'], cond=None, rcond=None,
    lower=True, check_finite=True):
    """ Returns true when input is positive-definite,
    via given method
    method options:
    'cholesky'
    'eigvals'
    'scipy'
    """

    A = np.atleast_2d(A)
    method = copy.deepcopy(methods)
    if 'all' in method:
        method = ['scipy', 'cholesky', 'eigvals']

    pd_check = True
    if 'cholesky' in method:
        try:
            _ = la.cholesky(A)
            pd_check = True and pd_check
        except la.LinAlgError:
            pd_check = False and pd_check

    if 'eigvals' in method:
        machine_epsilon = np.finfo(np.float64).eps
        threshold = np.sqrt(machine_epsilon)
        if np.all(la.eigvals(A) >= threshold):
            pd_check = True and pd_check
        else:
            pd_check = False and pd_check

#        pd_check =  np.all(la.eigvals(A) >= threshold) and pd_check

    if 'scipy' in method:
        s, u = scipy.linalg.eigh(A, lower=lower, check_finite=check_finite)

        eps = _eigvalsh_to_eps(s, cond, rcond)

        if np.min(s) > - eps:
            pd_check = True and pd_check
        else:
            pd_check = False and pd_check

#        pd_check = np.min(s) > -eps and pd_check
# want to create type 'bool' instead of type 'numpy.bool_' so we
# can add variables in posterior_update.py
    return pd_check



def get_near_psd(A, min_norm='Frob', change_measure='machine_eps'):
    """ Consistent with Springer Series in Operations Research
    book on 'Numerical Optimization'. Secion 3.4
    Eigenvalue Modification
    Eqs. (3.40), (3.41) and (3.43)
    https://link.springer.com/book/10.1007/978-0-387-40065-5
    """
    # make symmetric
    A_symm = (A + A.T)/2
    A_symm_bk = A_symm.copy()

    if isPD(A_symm, methods=['scipy', 'eigvals']):
        return A_symm

    # check for negative eigenvalues
    eigval, eigvec = np.linalg.eig(A_symm)
    D = len(eigval)

    if change_measure == 'machine_eps':
        machine_epsilon = np.finfo(np.float64).eps
        delta = np.sqrt(machine_epsilon)
    elif change_measure == 'flip':
        delta = 1

    if min_norm is None:
        if change_measure == 'machine_eps':
            eigval[eigval < 0] = delta
        elif change_measure == 'flip':
            eigval[eigval < 0] *= -1

    elif min_norm == 'Frob':
        # correction matrix Delta_A with minimum Frobenius norm
        tau = np.zeros_like(eigval)
        tau[eigval >= delta] = 0
        tau[eigval < delta] = delta - eigval[eigval < delta]
        Delta_A = eigvec.dot(np.diag(tau)).dot(eigvec.T)
        A_symm += Delta_A
        A_symm2 = eigvec.dot(np.diag(eigval + tau)).dot(eigvec.T)
        frob_norm = np.linalg.norm(A_symm)
        eigval, eigvec = np.linalg.eig(A_symm)
#        assert min(eigval) >= delta

    elif min_norm == 'euc':
        # correction matrix Delta_A with minimum Euclidean norm
        tau = max(0, delta - min(eigval))
        Delta_A = np.eye(D) * tau
        zeros = np.zeros((D))
        A_symm += Delta_A
        euc_norm = np.linalg.norm(zeros, np.diag(Delta_A))
        eigval, eigvec = np.linalg.eig(A_symm)
#        assert min(eigval) >= delta

    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)



# DTS: these should not be defined within this function
def compute_pred_cov_deriv(pred_pts_t, build_pts_t, pred_cov, length_scales):
    npred_points = pred_pts_t.shape[0]
    nbuild_points = build_pts_t.shape[0]
    ndims = build_pts_t.shape[1]
    pred_scaled_dists = np.zeros((npred_points, nbuild_points, ndims))
    for kk in range(ndims):
        # compute mixed (i.e. prediction - build) distances
        # and divide by length scale hyperparameters
        pred_scaled_dists[:, :, kk] = cdist(pred_pts_t[:, kk].reshape(-1, 1),
            build_pts_t[:, kk].reshape(-1, 1),
            lambda u, v : u - v) / length_scales[kk]**2
    return np.einsum("ijk, ij -> ijk", -pred_scaled_dists, pred_cov)



# DTS: these should not be defined within this function
def compute_gp_grad_mu(GP, input_scaler, output_scaler,
        alpha_vec, build_pts_t, pred_pts_t):
    ndims = build_pts_t.shape[1]
    pred_cov = GP.kernel_(pred_pts_t, build_pts_t)
    length_scale_hyperparams = np.exp(GP.kernel_.theta[1:]) # [scaling, length scale params] - sklearn stores the log of theta
    pred_deriv = compute_pred_cov_deriv(pred_pts_t, build_pts_t, pred_cov,
        length_scale_hyperparams)

    # scaled gradient
    grad_mu = np.einsum("ijk, j -> ik", pred_deriv, alpha_vec)
    for kk in range(ndims):
        # this assumes targets are 1D
        scale_factor = output_scaler.scale_[0] / input_scaler.scale_[kk]
        grad_mu[:, kk] *= scale_factor
    return grad_mu



def analytical_GP_gradient(pred_pts, GP_dict, input_scaler, build_pts):

    GP = GP_dict['GP']
    output_scaler = GP_dict['output_scaler']
    build_target = GP_dict['ys']

    pred_pts = np.atleast_2d(pred_pts)

    if input_scaler is not None:
        build_pts_t = input_scaler.transform(build_pts)
        pred_pts_t = input_scaler.transform(pred_pts)
    else:
        build_pts_t = build_pts
        pred_pts_t = pred_pts

    if output_scaler is not None:
        build_target_t = output_scaler.transform(build_target)
    else:
        build_target_t = build_target

    alpha_vec = cho_solve((GP.L_, True), build_target_t).squeeze()
    grad_mu = compute_gp_grad_mu(GP, input_scaler, output_scaler,
       alpha_vec, build_pts_t, pred_pts_t)

    return grad_mu



def compute_centered_fd_epsilon(x, order=2, approx_type="grad",
    scale_eps=True):

   machine_epsilon = np.finfo(np.float64).eps

   if approx_type == "grad":
       if order == 2:
           factor_machine_epsilon = np.cbrt(machine_epsilon)
       elif order == 4:
           factor_machine_epsilon = machine_epsilon ** 0.25
       else:
           raise ValueError("FD order not yet implemented for central difference")

   elif approx_type == "hessian":
       if order == 2:
           factor_machine_epsilon = (machine_epsilon)**(0.25)
       elif order == 4:
           factor_machine_epsilon = (machine_epsilon)**(0.2)
       else:
           raise ValueError("FD order not yet implemented for centered difference")

   else:
       raise ValueError("FD derivative approx type not implemented")

   fd_epsilon = np.ones_like(x) * factor_machine_epsilon
   it = np.nditer(x, flags=["multi_index"])

   if scale_eps:
       for xval in it:
    #       scale = -1. if xval < 0. else 1.
           scale = 1
           fd_epsilon[it.multi_index] *= max(abs(xval), 1.) * scale
    #       fd_epsilon[it.multi_index] *= (1 + abs(xval))
   return fd_epsilon



def compute_fd_gradient(x, func, fd_eps="auto", order=2, scale_eps=True):
    """
    Compute the gradient vector at points x

    Parameters:
        - x: numpy array of size num_points by dim
        - func: function that takes numpy arrays as input

    Optional:
        - fd_eps : automatically determined using cbrt
        of machine precision or manually specified with a scalar
        value

    Returns:
        a matrix with size num_points x dim that contains gradient
        values at different points

    """

    assert len(x.shape) <= 2
    if len(x.shape) == 1:
        x = np.atleast_2d(x)
    num_points = x.shape[0]
    dim = x.shape[1]
    grad = np.empty_like(x)

    if fd_eps == "auto":
       fd_epsilon = compute_centered_fd_epsilon(x, order=order, scale_eps=scale_eps)
    else:
        assert np.isscalar(fd_eps) or fd_eps.shape[1] == dim
        fd_epsilon = np.ones_like(x) * fd_eps

    for ii in range(dim):

        perturb = fd_epsilon[:, ii].squeeze()

        if order == 2:

            x_plus = x.copy()
            x_plus[:, ii] += perturb
            f_plus = func(x_plus).squeeze()

            x_minus = x.copy()
            x_minus[:, ii] -= perturb
            f_minus = func(x_minus).squeeze()

            grad[:, ii] = (f_plus - f_minus) / (2. * perturb)

        elif order == 4:

            x_plus = x.copy()
            x_plus[:, ii] += perturb
            f_plus = func(x_plus).squeeze()

            x_plus2 = x.copy()
            x_plus2[:, ii] += perturb * 2.
            f_plus2 = func(x_plus2).squeeze()

            x_minus = x.copy()
            x_minus[:, ii] -= perturb
            f_minus = func(x_minus).squeeze()

            x_minus2 = x.copy()
            x_minus2[:, ii] -= perturb * 2.
            f_minus2 = func(x_minus2).squeeze()

            grad[:, ii] = (-f_plus2 + 8. * f_plus - 8. * f_minus + \
                f_minus2) / (12. * perturb)

    return grad



def fd_check_components(ref_val, x, func,
    epsilons=np.logspace(-1, -13, 13), test_type="grad",
    order=2):

    """ compare finite difference approxoimation of gradient or
    hessian to reference value. The absolute error is returned. """

    x = np.atleast_2d(x)
    assert x.shape[0] == 1 # single point only
    dim = x.shape[1]

    num_eps = len(epsilons)

    if test_type == 'grad':
        fd_method = compute_fd_gradient
        fd_approx_err = np.zeros((num_eps, dim))
        ref_val = ref_val.squeeze()
    elif test_type == 'hessian':
        #fd_check = compute_fd_hessian
        fd_method = compute_fd_hessian
        unique_idx = np.triu_indices(dim)
        num_unique_entries = dim * (dim + 1) // 2
        fd_approx_err = np.zeros((num_eps, num_unique_entries))
        ref_val = np.atleast_2d(ref_val)
    else:
       raise ValueError("fd check test type not valid")

    for idx, fd_eps in enumerate(epsilons):
        fd_approx = fd_method(x, func, fd_eps=fd_eps, order=order)
        if test_type == "grad":
            fd_approx_err[idx, :] = np.abs(ref_val - fd_approx)
        elif test_type == "hessian":
            fd_approx_err[idx, :] = np.abs(ref_val[unique_idx] \
                - fd_approx[unique_idx])

    return fd_approx_err



def fd_check_directional(ref_deriv, x, func, direction="random",
    seed=44, epsilons=np.logspace(0, -12, 13),
    test_type="grad", order=2):

    x = np.atleast_2d(x)
    ref_deriv = np.atleast_2d(ref_deriv)
    dim = x.shape[1]
    assert x.shape[0] == 1
    if test_type == "grad":
        assert x.shape == ref_deriv.shape
    elif test_type == "hessian":
        assert ref_deriv.shape[0] == dim
        assert ref_deriv.shape[1] == dim
    else:
        raise ValueError("derivative type not implementd")

    if direction == "random":
        np.random.seed(seed)
        d = np.atleast_2d(np.random.rand(dim) - 0.5)
    else:
        d = np.atleast_2d(direction)
        assert d.shape[0] == 1
        assert d.shape[1] == dim

    num_eps = len(epsilons)
    fd_error = np.empty(num_eps)

    if test_type == "grad":
        deriv_product = (d @ ref_deriv.T).squeeze()
    elif test_type == "hessian":
        deriv_product = (d @ ref_deriv @ d.T).squeeze()
        f_ref = func(x).squeeze()

    for idx, fd_epsilon in enumerate(epsilons):
        if order == 2:
            x_plus = x.copy()
            x_minus = x.copy()
            perturb = fd_epsilon * d
            x_plus += perturb
            x_minus -= perturb
            f_plus = func(x_plus).squeeze()
            f_minus = func(x_minus).squeeze()
            if test_type == "grad":
                fd_approx = (f_plus - f_minus) / (2. * fd_epsilon)
            elif test_type == "hessian":
                fd_approx = (f_plus + f_minus - 2. * f_ref) / fd_epsilon**2
        elif order == 4:
            x_plus = x.copy()
            x_minus = x.copy()
            perturb = fd_epsilon * d
            x_plus += perturb
            x_minus -= perturb
            f_plus = func(x_plus).squeeze()
            f_minus = func(x_minus).squeeze()

            x_plus2 = x.copy()
            x_minus2 = x.copy()
            x_plus2 += perturb * 2.
            x_minus2 -= perturb * 2.
            f_plus2 = func(x_plus2).squeeze()
            f_minus2 = func(x_minus2).squeeze()
            if test_type == "grad":
                fd_approx = (- f_plus2 + 8. * f_plus - 8. * f_minus\
                    + f_minus2) / (12. * fd_epsilon)
            elif test_type == "hessian":
                fd_approx = (f_plus + f_minus - 2. * f_ref) / fd_epsilon**2
        fd_error[idx] = np.abs(fd_approx - deriv_product)

    return fd_error


def compute_jacobian(funs, x_val, fd_eps=1e-6, order=2, scale_eps=True, **kwargs):
    """ Compute the Jacobian of funs

        Parameters:
        - funs: list of functions take parameter as input and
          product QOI as output (may be a partial function)

        - x_val: ndarry (1 x ndim)

        - grad (optional): list of function gradients, if this
          kwarg is passed, then the analytical gradient is
          calculated, otherwise, the FD gradient is calculated

        ** all function gradients calculated with same method
              (FD or analytical)
    """

    x_val = np.atleast_2d(x_val)
    nfuns = len(funs)
    ndim = x_val.shape[1]

    # determine if analytical expressions provided for gradients
    for arg in kwargs.keys():
        if arg == 'grad':
            grads = kwargs[arg]

    jac = np.zeros((nfuns, ndim))
    for i, fun in enumerate(funs):
        if 'grads' not in locals():
            jac[i,:] = compute_fd_gradient(x_val, fun, fd_eps=fd_eps, order=order,
                scale_eps=scale_eps)

        else:
            grad_fun = grads[i]
            jac[i,:] = grad_fun(x_val)

    return jac



def compute_fd_hessian(x, func, fd_eps="auto", order=2, scale_eps=True):
    # add test for scaled input case
    """ Compute the Hessian of a function at single point x

    Parameters:
        - x: numpy array for point x
        - func: function

    Returns:
        Hessian
    """

    assert len(x.shape) <= 2
    if len(x.shape) == 1:
        x = np.atleast_2d(x)
    assert x.shape[0] == 1
    dim = x.shape[1]
    hessian = np.empty((dim, dim))

    f_ref = func(x).squeeze()

    if isinstance(fd_eps, str):
       if fd_eps == 'auto':
           fd_epsilon = compute_centered_fd_epsilon(x, order=order,
               approx_type="hessian", scale_eps=scale_eps)
       else:
           raise ValueError('finite different pertubation method not yet implemented')
    else:
        assert np.isscalar(fd_eps) or fd_eps.shape[1] == dim
        fd_epsilon = np.ones_like(x) * fd_eps

    for ii in range(dim):
        for jj in range(dim):

            if order == 2:
                if ii == jj:
                    perturb = fd_epsilon[:, ii].squeeze()

                    x_p = x.copy()
                    x_p[:, ii] += perturb
                    f_p = func(x_p).squeeze()

                    x_m = x.copy()
                    x_m[:, ii] -= perturb
                    f_m = func(x_m).squeeze()

                    hessian[ii, ii] = (f_p + f_m - 2. * f_ref) \
                        / perturb**2.

                elif ii < jj:
                    perturb_ii = fd_epsilon[:, ii].squeeze()
                    perturb_jj = fd_epsilon[:, jj].squeeze()

                    x_pp = x.copy()
                    x_pp[:, ii] += perturb_ii
                    x_pp[:, jj] += perturb_jj
                    f_pp = func(x_pp).squeeze()

                    x_mm = x.copy()
                    x_mm[:, ii] -= perturb_ii
                    x_mm[:, jj] -= perturb_jj
                    f_mm = func(x_mm).squeeze()

                    x_pm = x.copy()
                    x_pm[:, ii] += perturb_ii
                    x_pm[:, jj] -= perturb_jj
                    f_pm = func(x_pm).squeeze()

                    x_mp = x.copy()
                    x_mp[:, ii] -= perturb_ii
                    x_mp[:, jj] += perturb_jj
                    f_mp = func(x_mp).squeeze()

                    hessian[ii, jj] = (f_pp + f_mm - f_pm - f_mp) \
                        / (4. * perturb_ii * perturb_jj)

                else:
                     hessian[ii, jj] = hessian[jj, ii]


            elif order == 4:
                if ii == jj:
                    perturb = fd_epsilon[:, ii].squeeze()

                    x_p = x.copy()
                    x_p[:, ii] += perturb
                    f_p = func(x_p).squeeze()

                    x_m = x.copy()
                    x_m[:, ii] -= perturb
                    f_m = func(x_m).squeeze()

                    x_pp = x.copy()
                    x_pp[:, ii] += perturb * 2.
                    f_pp = func(x_pp).squeeze()

                    x_mm = x.copy()
                    x_mm[:, ii] -= perturb * 2.
                    f_mm = func(x_mm).squeeze()

                    hessian[ii, ii] = (-f_pp + 16. * f_p - 30. * f_ref \
                        + 16. * f_m - f_mm) / (12. * perturb**2.)



                elif ii < jj:
                    perturb_ii = fd_epsilon[:, ii].squeeze()
                    perturb_jj = fd_epsilon[:, jj].squeeze()
                    idx = [ii, jj]

                    x_pp_pp = x.copy()
                    adj = [perturb_ii * 2., perturb_jj * 2.]
                    x_pp_pp[:, idx] += adj
                    f_pp_pp = func(x_pp_pp).squeeze()

                    x_pp_p = x.copy()
                    adj = [perturb_ii * 2., perturb_jj]
                    x_pp_p[:, idx] += adj
                    f_pp_p = func(x_pp_p).squeeze()

                    x_pp_m = x.copy()
                    adj = [perturb_ii * 2., -perturb_jj]
                    x_pp_m[:, idx] += adj
                    f_pp_m = func(x_pp_m).squeeze()

                    x_pp_mm = x.copy()
                    adj = [perturb_ii * 2., -perturb_jj * 2.]
                    x_pp_mm[:, idx] += adj
                    f_pp_mm = func(x_pp_mm).squeeze()

                    x_p_pp = x.copy()
                    adj = [perturb_ii, perturb_jj * 2.]
                    x_p_pp[:, idx] += adj
                    f_p_pp = func(x_p_pp).squeeze()

                    x_p_p = x.copy()
                    adj = [perturb_ii, perturb_jj]
                    x_p_p[:, idx] += adj
                    f_p_p = func(x_p_p).squeeze()

                    x_p_m = x.copy()
                    adj = [perturb_ii, -perturb_jj]
                    x_p_m[:, idx] += adj
                    f_p_m = func(x_p_m).squeeze()

                    x_p_mm = x.copy()
                    adj = [perturb_ii, -perturb_jj * 2.]
                    x_p_mm[:, idx] += adj
                    f_p_mm = func(x_p_mm).squeeze()

                    x_m_pp = x.copy()
                    adj = [-perturb_ii, perturb_jj * 2.]
                    x_m_pp[:, idx] += adj
                    f_m_pp = func(x_m_pp).squeeze()

                    x_m_p = x.copy()
                    adj = [-perturb_ii, perturb_jj]
                    x_m_p[:, idx] += adj
                    f_m_p = func(x_m_p).squeeze()

                    x_m_m = x.copy()
                    adj = [-perturb_ii, -perturb_jj]
                    x_m_m[:, idx] += adj
                    f_m_m = func(x_m_m).squeeze()

                    x_m_mm = x.copy()
                    adj = [-perturb_ii, -perturb_jj * 2.]
                    x_m_mm[:, idx] += adj
                    f_m_mm = func(x_m_mm).squeeze()

                    x_mm_pp = x.copy()
                    adj = [-perturb_ii * 2., perturb_jj * 2.]
                    x_mm_pp[:, idx] += adj
                    f_mm_pp = func(x_mm_pp).squeeze()

                    x_mm_p = x.copy()
                    adj = [-perturb_ii * 2., perturb_jj]
                    x_mm_p[:, idx] += adj
                    f_mm_p = func(x_mm_p).squeeze()

                    x_mm_m = x.copy()
                    adj = [-perturb_ii * 2., -perturb_jj]
                    x_mm_m[:, idx] += adj
                    f_mm_m = func(x_mm_m).squeeze()

                    x_mm_mm= x.copy()
                    adj = [-perturb_ii * 2., -perturb_jj * 2.]
                    x_mm_mm[:, idx] += adj
                    f_mm_mm = func(x_mm_mm).squeeze()

                    term_1 = - (-f_pp_pp + 8. * f_pp_p \
                         - 8. * f_pp_m + f_pp_mm)
                    term_2 = 8. * (-f_p_pp + 8. * f_p_p \
                         - 8. * f_p_m + f_p_mm)
                    term_3 = - 8. * (- f_m_pp + 8. * f_m_p \
                         - 8. * f_m_m + f_m_mm)
                    term_4 = (- f_mm_pp + 8. * f_mm_p \
                         - 8. * f_mm_m + f_mm_mm)
                    hessian[ii, jj] = (term_1 + term_2 + term_3 + term_4) \
                        / (144. * perturb_ii * perturb_jj)
                else:
                    hessian[ii, jj] = hessian[jj, ii]


    return hessian


def compute_laplace_approx(func, map_point, fd_perturbation=1.0e-6):
    # approximate the Hessian at the map point
    # DTS: fd_perturbation not even used here?

    hessian = compute_fd_hessian(func, map_point)

    det_tol = 1.0e-10
    if np.linalg.det(hessian) < det_tol:
        count = 0
        while np.linalg.det(hessian) < 0:
            hessian += np.eye(2) * det_tol
            count += 1

    cov = np.linalg.inv(hessian)
    det_hessian = np.linalg.det(hessian)
    return map_point, cov, det_hessian


def is_picklable(obj):

    try:
        pickle.dump(obj)
    except Exception:
        return False
    return True
