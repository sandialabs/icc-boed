import numpy as np
from abc import abstractmethod
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.decomposition import PCA

import matplotlib
import matplotlib.pyplot as plt
import importlib


#from pyapprox import surrogates
#from pyapprox.surrogates.interp.indexing import compute_hyperbolic_indices

class GaussianProcess:

    def __init__(self, construct_pca_basis=False,
        input_scaling=False, output_scaling=False,
        separate_gps=False, *args, **kwargs):

        self.construct_pca_basis = construct_pca_basis
        self.input_scaling = input_scaling
        self.output_scaling = output_scaling
        self.separate_gps = separate_gps
        self.nsurrogates = -1
        self.surrogate_type = 'gaussian_process'

        self.nrestarts = 0
        self.alpha = 1.0e-10
        self.normalize_y = False
        self.kernel_type = "SE"
        self.random_state = None
        self.output_scaler_with_std = True
        for kwarg in kwargs:
            if kwarg == 'n_restarts_optimizer':
                self.nrestarts = kwargs[kwarg]
            if kwarg == 'nmodes':
                self.nmodes = kwargs[kwarg]
            if kwarg == 'alpha':
                self.alpha = kwargs[kwarg]
            if kwarg == 'normalize_y':
                self.normalize_y = kwargs[kwarg]
            if kwarg == 'kernel_type':
                self.kernel_type = kwargs[kwarg]
            if kwarg == 'random_state':
                self.random_state = kwargs[kwarg]
            if kwarg == 'output_scaler_with_std':
                self.output_scaler_with_std = kwargs[kwarg]



    def compute_pca_basis(self, X):
        # X is (nsamples x nfeatures) array
        # optionally: X is (nsamples x nfeatures x nqois) array
        # then, a different basis is calculated for each qoi

        assert X.ndim >= 2 and X.ndim <= 3
        if X.ndim == 2:
            X = X[..., np.newaxis]
        assert X.ndim == 3

        if not hasattr(self, "nmodes"):
            self.nmodes = min(X.shape[0], X.shape[1]) - 1

        self.pca_features = X
        self.npca_samples = X.shape[0]
        self.nbases = X.shape[2]

        self.pca = [None] * self.nbases
        self.pca_coeffs = np.zeros((self.npca_samples, self.nmodes,
            self.nbases))

        for ii in np.arange(self.nbases):
            self.pca[ii] = PCA(n_components=self.nmodes, svd_solver="full")
            self.pca_coeffs[..., ii] = self.pca[ii].fit_transform(\
                X[..., ii])


            if False:
                plot_modes = np.arange(2, 16)
                nn = len(plot_modes)
                exp_var = np.zeros(nn)
                for it in np.arange(nn):
                    pca_test = PCA(n_components=plot_modes[it], svd_solver="full")
                    coeffs = pca_test.fit_transform(X[..., ii])
                    exp_var[it] = pca_test.explained_variance_ratio_.sum()
                print(exp_var)
                plt.figure(figsize=(10,8))
                plt.plot(plot_modes, exp_var, color='r', marker='d', markersize=18, linestyle='None')
                plt.xlabel('Number of Retained Components', fontsize=30)
                plt.ylabel('Explained Variance', fontsize=30)
                plt.xticks(np.arange(2, nn+1, step=2), fontsize=30)
                plt.yticks(fontsize=30)
                plt.tight_layout()
                plt.savefig(f'figures/determine_number_of_PCA_modes/explained_variance_{ii}.png')
 #               plt.show()
                plt.close()

                plt.figure(figsize=(10,8))
                pca_test = PCA()
                pca_test.fit_transform(X[..., ii])
                sing_vals = pca_test.singular_values_
                max_val = sing_vals.max()
                norm_vals = sing_vals/max_val
                nvals = len(sing_vals)
                plt.plot(np.arange(self.nmodes), norm_vals[:self.nmodes], color='r', marker='.', markersize=18, linestyle='None')
                plt.plot(np.arange(self.nmodes, nvals-1), norm_vals[self.nmodes:-1], color='b', marker='.', markersize=18, linestyle='None')
                plt.yscale('log')
                plt.xlabel('Mode Index', fontsize=28)
                plt.ylabel('Normalized Singular Values', fontsize=28)
                plt.xticks(np.arange(nvals+1, step=100)[1:], fontsize=28)
                plt.yticks(fontsize=28)
                plt.tight_layout()
#                plt.show()
                plt.savefig(f'figures/determine_number_of_PCA_modes/singular_values_{ii}.png')
                plt.close()

            if False:
                explained_variance = self.pca[ii].explained_variance_ratio_.sum()
                pca_fit = self.pca[ii].inverse_transform(self.pca_coeffs[...,ii])
                plt.figure()
                if self.output_scaling:
                    X_scaled = self.output_scaler[ii].inverse_transform(X[..., ii])
                    pca_fit = self.output_scaler[ii].inverse_transform(pca_fit)
                    plt.plot(X_scaled[0, :].T, 'r*', ms=3, label='Training Data')
                    plt.plot(X_scaled[::10, :].T, 'r*', ms=3)
                else:
                    plt.plot(X[0, :, ii].T, 'r*', ms=3, label='Training Data')
                    plt.plot(X[::10, :, ii].T, 'r*', ms=3)
                plt.plot(pca_fit[0, :].T, 'bo', ms=1, label='PCA Fit')
                plt.plot(pca_fit[::10, :].T, 'bo', ms=1)
                plt.title(f'PCA Fit: {self.nmodes} Modes, Explained Variance {explained_variance}')
                plt.legend()
                plt.savefig(f'figures/determine_number_of_PCA_modes/pca_fit.png')

                plt.figure()
                singular_values = self.pca[ii].singular_values_
                plt.plot(singular_values, 'bo')
                plt.title('Singular Values')
                plt.savefig(f'figures/determine_number_of_PCA_modes/singular_values_{ii}.png')
#                plt.show()


    def build(self, X, Y):
        assert X.ndim == 2
        self.nsamples = X.shape[0]
        self.nfeatures = X.shape[1]
        assert Y.ndim >= 2 & Y.ndim <= 3
        if Y.ndim == 2:
            Y = Y[..., np.newaxis]
        assert Y.ndim == 3
        assert self.nsamples == Y.shape[0]
        self.ntargets = Y.shape[1]
        self.nqoi = Y.shape[2]

        gp_train_features = X
        if self.input_scaling:
            self.input_scaler = StandardScaler()
            scaled_features = self.input_scaler.fit_transform(X)
            gp_train_features = scaled_features

        gp_train_targets = Y
        if self.output_scaling:
            self.output_scaler = [None] * self.nqoi
            scaled_targets = np.zeros_like(Y)
            for ii in np.arange(self.nqoi):
                self.output_scaler[ii] = StandardScaler(with_std=self.output_scaler_with_std)
                scaled_targets[..., ii] = self.output_scaler[ii]\
                    .fit_transform(Y[..., ii])

                if False:
                    test_scaler = StandardScaler(with_std=False)
                    test_scaled_targets = test_scaler.fit_transform(Y[...,ii])
                    import matplotlib.pyplot as plt
                    import matplotlib
                    fig, ax = plt.subplots(3,3)
                    ax[0,0].plot(Y[...,ii].T, 'r*')
                    ax[0,1].plot(self.scaled_targets[...,ii].T, 'b*')
                    ax[0,2].plot(test_scaled_targets.T, 'g*')
                    ax[1,0].plot(Y[...,ii].mean(axis=0), 'r*')
                    ax[1,1].plot(self.scaled_targets[...,ii].mean(axis=0), 'b*')
                    ax[1,2].plot(test_scaled_targets.mean(axis=0), 'g*')
                    ax[2,0].plot(Y[...,ii].var(axis=0), 'r*')
                    ax[2,1].plot(self.scaled_targets[...,ii].var(axis=0), 'b*')
                    ax[2,2].plot(test_scaled_targets.var(axis=0), 'g*')
                    ax[0,0].set_ylabel('data')
                    ax[1,0].set_ylabel('mean')
                    ax[2,0].set_ylabel('var')
                    ax[0,0].set_title('original data')
                    ax[0,1].set_title('0 mean unit variance')
                    ax[0,2].set_title('0 mean')

                    fig, ax = plt.subplots(1,3)
                    ax[0].plot(Y[100, ...,ii].T, 'r*')
                    ax[1].plot(self.scaled_targets[100,...,ii].T, 'b*')
                    ax[2].plot(test_scaled_targets[100,...].T, 'g*')
                    ax[0].set_title('data')
                    ax[1].set_title('0 mean unit variance')
                    ax[2].set_title('0 mean')
                    plt.show()

            gp_train_targets = scaled_targets

        if self.construct_pca_basis:
            self.compute_pca_basis(gp_train_targets)
            gp_train_targets = self.pca_coeffs

        self.ngp_targets = gp_train_targets.shape[1]

        if self.separate_gps:
            self.surrogates = [[None] * self.ngp_targets \
                for i in np.arange(self.nqoi)]
        else:
            self.surrogates = [[None] for i in np.arange(self.nqoi)]
        self.nsurrogates = len(self.surrogates[0])

        if self.kernel_type == "SE":
            self.kernels = [C(1.0, constant_value_bounds=(1e-2, 1e2))\
                * RBF(np.ones(self.nfeatures), length_scale_bounds=(1e-2, 1e2))
                for ii in range(self.nsurrogates)]
        elif self.kernel_type == "Matern":
            self.kernels = [C(1.0, constant_value_bounds=(1e-2, 1e2))\
                * Matern(np.ones(self.nfeatures), length_scale_bounds=(1e-2, 1e2), nu=1.5) for ii in range(self.nsurrogates)]
        else:
            raise ValueError("kernel type not implemented")

        for ii in range(self.nqoi):
            for jj in range(self.nsurrogates):
                self.surrogates[ii][jj] = \
                    GaussianProcessRegressor(
                        kernel=self.kernels[jj],
                        n_restarts_optimizer=self.nrestarts,
                        alpha=self.alpha,
                        random_state=self.random_state,
                        normalize_y=self.normalize_y)
                if self.separate_gps:
                    self.surrogates[ii][jj].fit(
                        gp_train_features,
                        np.atleast_2d(gp_train_targets[:, jj, ii]).T)
                else:
                    self.surrogates[ii][jj].fit(
                        gp_train_features,
                        np.atleast_2d(gp_train_targets[:, :, ii]))



    def value(self, X, return_pca=False):

        X = np.atleast_2d(X)
        assert X.ndim == 2
        assert X.shape[1] == self.nfeatures
        npred_pts = X.shape[0]

        if return_pca:
            pred = np.zeros((npred_pts, self.ngp_targets, self.nqoi))
        else:
            pred = np.zeros((npred_pts, self.ntargets, self.nqoi))

        pred_pts = X.copy()
        if self.input_scaling:
            pred_pts = self.input_scaler.transform(X)

        for ii in np.arange(self.nqoi):
            gp_pred = np.zeros((npred_pts, self.ngp_targets))

            if self.separate_gps:
                for jj in np.arange(self.nsurrogates):
                    gp_pred[:, jj] = self.surrogates[ii][jj].\
                        predict(pred_pts).squeeze()
            else:
                if self.ngp_targets > 1:
                    gp_pred = self.surrogates[ii][0].\
                        predict(pred_pts).squeeze()
                else:
                    gp_pred = np.atleast_2d(self.surrogates[ii][0].\
                        predict(pred_pts).squeeze()).T

            if self.construct_pca_basis and not return_pca:
                gp_pred = self.pca[ii].\
                    inverse_transform(gp_pred.squeeze())

            if self.output_scaling and not return_pca:
                gp_pred = np.atleast_2d(gp_pred)
                pred[..., ii] = self.output_scaler[ii].\
                    inverse_transform(gp_pred)
            else:
                pred[..., ii] = gp_pred

        return pred



    def error(self, X, Y, type="mse", multioutput="raw_values",
        pca=False):

        assert X.ndim == 2
        assert Y.ndim >= 2 & Y.ndim <= 3
        if Y.ndim == 2:
            Y = Y[..., np.newaxis]

        if pca:
            pred_pca_coefs = self.value(X, return_pca=True)
            Y_pca_coefs = np.zeros((Y.shape[0], self.nmodes, self.nqoi))
            if self.output_scaling:
                Y_scaled = np.zeros_like(Y)
                for ii in np.arange(self.nqoi):
                    Y_scaled[..., ii] = self.output_scaler[ii]\
                        .transform(Y[..., ii])
                    Y_pca_coefs[..., ii] = self.pca[ii]\
                        .transform(Y_scaled[..., ii])
            else:
                Y_pca_coefs[..., ii] = self.pca[ii]\
                    .transform(Y[..., ii])
            true_val = Y_pca_coefs
            pred_val = pred_pca_coefs

        else:
            pred_val = self.value(X)
            true_val = Y

        error = [None] * self.nqoi

        for ii in np.arange(self.nqoi):
            if type == "mse":
                error[ii] = mean_squared_error(
                    true_val[..., ii].T, pred_val[..., ii].T,
                    multioutput=multioutput)
            elif type == "mae":
                error[ii] = mean_absolute_error(
                    true_val[..., ii].T, pred_val[..., ii].T,
                    multioutput=multioutput)
            elif type == "mape":
                error[ii] = mean_absolute_percentage_error(
                    true_val[..., ii].T, pred_val[..., ii].T,
                    multioutput=multioutput)

        return error



    def score(self, X, Y):

        assert X.ndim == 2
        assert Y.ndim >= 2 & Y.ndim <= 3
        if Y.ndim == 2:
            Y = Y[..., np.newaxis]

        pred = self.value(X)
        score = [None] * self.nqoi

        for ii in np.arange(self.nqoi):
            u = ((Y[..., ii] - pred[..., ii])**2).sum()
            v = ((Y[..., ii] - Y[..., ii].mean())**2).sum()
            score[ii] = 1 - (u/v)
        return score



    def gradient(self):
        ...

    def hessian(self):
        ...


class PolynomialChaosExpansion:


    def __init__(self, verbosity=False, construct_pca_basis=False,
        input_scaling=False, output_scaling=False,
        separate_pce=False, *args, **kwargs):

        pyapprox = importlib.import_module("pyapprox")
        surrogates = pyapprox.surrogates
        compute_hyperbolic_indices = surrogates.interp.indexing.compute_hyperbolic_indices

        if False:
            try:
                from pyapprox import surrogates
            except ImportError:
                self.pya_surrogates = None
            else:
                self.pya_surrogates = surrogates


            try:
                from pyapprox.surrogates.interp.indexing import compute_hyperbolic_indices
            except ImportError:
                self.pya_compute_hyperbolic_indices = None
            else:
                self.pya_compute_hyperbolic_indices = compute_hyperbolic_indices

        self.construct_pca_basis = construct_pca_basis
        self.input_scaling = input_scaling
        self.output_scaling = output_scaling
        self.separate_pce = separate_pce
        self.nsurrogates = -1
        self.surrogate_type = 'polynomial_chaos_expansion'

        # basis_type: "fixed", "hyperbolic_cross", "expanding_basis"
        self.basis_type = "fixed"
        self.variable = None
        self.output_scaler_with_std = False
        for kwarg in kwargs:
            if kwarg == 'nmodes':
                self.nmodes = kwargs[kwarg]
            if kwarg == 'basis_type':
                self.basis_type = kwargs[kwarg]
            if kwarg == 'variable':
                self.variable = kwargs[kwarg]
            if kwarg == 'output_scaler_with_std':
                self.output_scaler_with_std = kwargs[kwarg]
        if self.variable is None:
            raise ValueError('must provide variable')

        if self.basis_type == "fixed":
            self.degree = 1
            self.hcross_strength = 1
            # solver_type: "lasso", "lars", "lasso_grad", "omp", "lstsq"
            self.solver_type = "lstsq"
            self.npars = None
            for kwarg in kwargs:
                if kwarg == 'degree':
                    self.degree = kwargs[kwarg]
                if kwarg == 'hcross_strength':
                    self.hcross_strength = kwargs[kwarg]
                if kwarg == 'solver_type':
                    self.solver_type = kwargs[kwarg]
                if kwarg == 'npars':
                    self.npars = kwargs[kwarg]
            if self.npars is None:
                raise ValueError('must provide npars')
            self.indices = compute_hyperbolic_indices(\
                self.npars, self.degree, self.hcross_strength)
            self.opts = {"verbosity": verbosity,
                "basis_type": self.basis_type,
                "variable": self.variable,
                "options":\
                {"linear_solver_options": {},
                "indices": self.indices,
                "solver_type": self.solver_type,
                "verbose": verbosity}}
        elif self.basis_type == "hyperbolic_cross":
            self.min_degree = 1
            self.max_degree = 20
            for kwarg in kwargs:
                if kwarg == "min_degree":
                    self.min_degree = kwargs[kwarg]
                if kwarg == "max_degree":
                    self.max_degree = kwargs[kwarg]
            self.opts = {"verbosity": verbosity,
                "basis_type": self.basis_type,
                "variable": self.variable,
                "options": {'min_degree': 1,
                "max_degree": self.max_degree,
                "verbose": verbosity}}
        elif self.basis_type == "expanding_basis":
            self.opts = {"verbosity": verbosity,
                "basis_type": self.basis_type,
                "variable": self.variable}
        else:
            raise ValueError('basis type must be "fixed", "hyperbolic_cross" or "expanding_basis"')



    def compute_pca_basis(self, X):
        # X is (nsamples x nfeatures) array
        # optionally: X is (nsamples x nfeatures x nqois) array
        # then, a different basis is calculated for each qoi

        assert X.ndim >= 2 and X.ndim <= 3
        if X.ndim == 2:
            X = X[..., np.newaxis]
        assert X.ndim == 3

        if not hasattr(self, "nmodes"):
            self.nmodes = min(X.shape[0], X.shape[1]) - 1

        self.pca_features = X
        self.npca_samples = X.shape[0]
        self.nbases = X.shape[2]

        self.pca = [None] * self.nbases
        self.pca_coeffs = np.zeros((self.npca_samples, self.nmodes,
            self.nbases))

        for ii in np.arange(self.nbases):
            self.pca[ii] = PCA(n_components=self.nmodes, svd_solver="full")
            self.pca_coeffs[..., ii] = self.pca[ii].fit_transform(\
                X[..., ii])



    def build(self, X, Y):
        pyapprox = importlib.import_module("pyapprox")
        surrogates = pyapprox.surrogates

        assert X.ndim == 2
        self.nsamples = X.shape[0]
        self.nfeatures = X.shape[1]
        assert Y.ndim >= 2 & Y.ndim <= 3
        if Y.ndim == 2:
            Y = Y[..., np.newaxis]
        assert Y.ndim == 3
        assert self.nsamples == Y.shape[0]
        self.ntargets = Y.shape[1]
        self.nqoi = Y.shape[2]

        pce_train_features = X
        if self.input_scaling:
            self.input_scaler = StandardScaler()
            scaled_features = self.input_scaler.fit_transform(X)
            pce_train_features = scaled_features

        pce_train_targets = Y
        if self.output_scaling:
            self.output_scaler = [None] * self.nqoi
            scaled_targets = np.zeros_like(Y)
            for ii in np.arange(self.nqoi):
                self.output_scaler[ii] = StandardScaler(with_std=self.output_scaler_with_std)
                scaled_targets[..., ii] = self.output_scaler[ii]\
                    .fit_transform(Y[..., ii])

            pce_train_targets = scaled_targets

        if self.construct_pca_basis:
            self.compute_pca_basis(pce_train_targets)
            pce_train_targets = self.pca_coeffs

        self.npce_targets = pce_train_targets.shape[1]

        if self.separate_pce:
            self.surrogates = [[None] * self.npce_targets \
                for i in np.arange(self.nqoi)]
        else:
            self.surrogates = [[None] for i in np.arange(self.nqoi)]
        self.nsurrogates = len(self.surrogates[0])

        for ii in range(self.nqoi):
            for jj in range(self.nsurrogates):
                if self.separate_pce:
                    self.surrogates[ii][jj] = \
                        surrogates.approximate(
                            pce_train_features.T,
                            np.atleast_2d(pce_train_targets[:, jj, ii]).T,
                            method='polynomial_chaos',
                            options=self.opts).approx
                else:
                    self.surrogates[ii][jj] =\
                        surrogates.approximate(
                            pce_train_features.T,
                            np.atleast_2d(pce_train_targets[:, :, ii]),
                            method='polynomial_chaos',
                            options=self.opts).approx



    def value(self, X, return_pca=False):

        X = np.atleast_2d(X)
        assert X.ndim == 2
        assert X.shape[1] == self.nfeatures
        npred_pts = X.shape[0]
        if return_pca:
            pred = np.zeros((npred_pts, self.npce_targets, self.nqoi))
        else:
            pred = np.zeros((npred_pts, self.ntargets, self.nqoi))

        pred_pts = X.copy()
        if self.input_scaling:
            pred_pts = self.input_scaler.transform(X)

        for ii in np.arange(self.nqoi):
            pce_pred = np.zeros((npred_pts, self.npce_targets))

            if self.separate_pce:
                for jj in np.arange(self.nsurrogates):
                    pce_pred[:, jj] = self.surrogates[ii][jj].\
                        value(pred_pts.T).squeeze()
            else:
                if self.npce_targets > 1:
                    pce_pred = self.surrogates[ii][0].\
                        value(pred_pts.T).squeeze()
                else:
                    pce_pred = np.atleast_2d(self.surrogates[ii][0].\
                        value(pred_pts.T).squeeze()).T

            if self.construct_pca_basis and not return_pca:
                pce_pred = self.pca[ii].\
                    inverse_transform(pce_pred.squeeze())

            if self.output_scaling and not return_pca:
                pce_pred = np.atleast_2d(pce_pred)
                pred[..., ii] = self.output_scaler[ii].\
                    inverse_transform(pce_pred)
            else:
                pred[..., ii] = pce_pred

        return pred



    def error(self, X, Y, type="mse", multioutput="raw_values",
        pca=False):

        assert X.ndim == 2
        assert Y.ndim >= 2 & Y.ndim <= 3
        if Y.ndim == 2:
            Y = Y[..., np.newaxis]

        if pca:
            pred_pca_coefs = self.value(X, return_pca=True)
            Y_pca_coefs = np.zeros((Y.shape[0], self.nmodes, self.nqoi))
            if self.output_scaling:
                Y_scaled = np.zeros_like(Y)
                for ii in np.arange(self.nqoi):
                    Y_scaled[..., ii] = self.output_scaler[ii]\
                        .transform(Y[..., ii])
                    Y_pca_coefs[..., ii] = self.pca[ii]\
                        .transform(Y_scaled[..., ii])
            else:
                Y_pca_coefs[..., ii] = self.pca[ii]\
                    .transform(Y[..., ii])
            true_val = Y_pca_coefs
            pred_val = pred_pca_coefs

        else:
            pred_val = self.value(X)
            true_val = Y

        error = [None] * self.nqoi

        for ii in np.arange(self.nqoi):
            if type == "mse":
                error[ii] = mean_squared_error(
                    true_val[..., ii].T, pred_val[..., ii].T,
                    multioutput=multioutput)
            elif type == "mae":
                error[ii] = mean_absolute_error(
                    true_val[..., ii].T, pred_val[..., ii].T,
                    multioutput=multioutput)
            elif type == "mape":
                error[ii] = mean_absolute_percentage_error(
                    true_val[..., ii].T, pred_val[..., ii].T,
                    multioutput=multioutput)

        return error



    def gradient(self):
        ...

    def hessian(self):
        ...

