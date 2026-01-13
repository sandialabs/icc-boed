import numpy as np
import itertools
from evaluate_models import select_strain_path, compute_qois
import matplotlib.pyplot as plt
from scipy.stats import qmc, sem
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import gp_tools as gpt
from qoi_models import mps_qois_fun
from functools import partial



def build_tree(nlevels, choice_list):

    tree_levels = np.arange(nlevels)
    tree = [None for i in np.arange(nlevels)]
    for level in tree_levels:
        tree[level] = \
            [p for p in itertools.product(\
            choice_list, repeat=level+1)]
    return tree



if __name__ == '__main__':

    np.random.seed(10)
    build_data_dir = "surrogate_data/"

    # initiate parameters
    pars = ["F", "G", "A", "n", "sigY"]
    D = len(pars)
    nFeatures = len(pars)

    lb = [0.3, 0.3, 10, 1e-6, 50]
    ub = [0.7, 0.7, 400., 100., 500]

    nTrainSamples = 500
    nTestSamples = 50
    alpha_val = 1e-8
    alpha_exp = 8

    # decision tree settings
    nlevels = 10
    choice_list = ['d1', 'd2']

    nInc = 100 # number of increments within each step
    epsInc = 0.02 #amount of strain applied each step
    num_qois = 3
    qoi_index = [0,1]
    qois = ['stress_11','stress_22','stress_33']
    nqoi = len(qoi_index) # number of qois

    nobs_plevel = 3 # number of observations per step
    obs_inds_by_level = [None for lev in range(nlevels)]
    obs_inds_bk = np.zeros(nlevels, dtype=int)
    for level in np.arange(nlevels):
        first_ob = level * (nInc - 1)
        last_ob = (level + 1) * nInc - (level + 1)
        obs_inds_bk[level] = last_ob
        obs_inds_by_level[level] = np.round(\
            np.linspace(first_ob, last_ob, nobs_plevel+1, dtype=int)[1:]).tolist()
    obs_inds = np.asarray([item for sublist in obs_inds_by_level for item in sublist])

    fmodel = partial(\
        mps_qois_fun,
        obs_inds=obs_inds,
        num_qois=num_qois,
        qoi_index=qoi_index,
        epsInc=epsInc,
        nInc=nInc,
        return_steps='last')
    testing_model = fmodel(\
        np.array((.55,.45, 150., 15., 200.)), design=['d1', 'd2', 'd1', 'd2', 'd1'],
        return_steps='last')

    if False:
        mps_obs_inds = np.arange(496)
        mps_out = fmodel(np.array((.55, .45, 150., 15., 200.)), design=['d1', 'd2', 'd1', 'd2', 'd1'],
            obs_inds=mps_obs_inds, return_steps='all')
        plt.figure()
        plt.plot(mps_obs_inds, mps_out[:, :, 0].squeeze(), 'r.', label='full response $\sigma_{11}$')
        plt.plot(mps_obs_inds, mps_out[:, :, 1].squeeze(), 'b.', label='full response $\sigma_{22}$')
        plt.plot(obs_inds, testing_model[:, :, 0].squeeze(), 'g.', markersize=10, label='observed')
        plt.plot(obs_inds, testing_model[:, :, 1].squeeze(), 'g.', markersize=10)
        plt.show()

   # build decision tree
    tree = build_tree(nlevels, choice_list)
    ntree_elem = 0
    for treeElem in tree:
        ntree_elem += len(treeElem)
    paths = [None] * ntree_elem

    path_idx = 0
    for level in np.arange(nlevels):
        for path in tree[level]:
            paths[path_idx] = list(path)
            path_idx += 1

    # generate strain paths for decision tree
    #strain_paths = tree_strain_paths(tree)

    train_features = gpt.get_nD_data(lb, ub, nTrainSamples,
        fmodel, return_targets=False)
    test_features = gpt.get_nD_data(lb, ub, nTestSamples,
        fmodel, return_targets=False)

    surrogate = {}
    surrogate['xs'] = train_features
    for p, path in enumerate(paths):

        path_name = '_'.join(path)
        surrogate[path_name] = {}
        path_level = len(path)

        train_targets = fmodel(train_features, path)
        test_targets = fmodel(test_features, path)

        nobs = test_targets.shape[1]
        nqois = test_targets.shape[2]
        for qoi in range(nqois):
            qoi_name = 'qoi_' + str(qoi)
            surrogate[path_name][qoi_name] = {}
            for ob in range(nobs):
                ob_name = 'observation_' + str(ob)
                print(f"On path {p}: {path_name},  Qoi: {qoi_name},  Observation: {ob_name}")
                surrogate[path_name][qoi_name][ob_name] = {}
                surrogate[path_name][qoi_name][ob_name]['GP'], surrogate['input_scaler'],\
                    surrogate[path_name][qoi_name][ob_name]['output_scaler'] =\
                    gpt.GP_build(train_features, train_targets[..., ob, qoi].reshape(-1, 1),
                    'SE', alpha_val=alpha_val, nrestarts=80)

#                build_surrogate(train_features, train_targets[:,:,qoi], alpha_val)

                surrogate[path_name][qoi_name][ob_name]['ys'] = train_targets[..., ob, qoi].reshape(-1, 1)

                gpt.GP_quality(surrogate[path_name][qoi_name][ob_name]['GP'], surrogate['input_scaler'],\
                test_features, train_features.reshape, test_targets[..., ob, qoi].reshape(-1, 1),\
                train_targets[..., ob, qoi].reshape(-1, 1))

    pickle.dump(surrogate, open(build_data_dir +\
        f"GP_all_paths_alpha_1e-{alpha_exp}_{nlevels}_levels_{D}_pars_{nobs_plevel}_obs_pstep_{epsInc}_strainInc", "wb"))

