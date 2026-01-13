import numpy as np
from gp_tools import eval_GP
from evaluate_models import compute_qois
from evaluate_models import select_strain_path
from scipy.stats import norm
import time
import pickle



def mps_experiment(design, theta_true,
    obs_inds, calc_qois, qoi_index,
    epsInc=None, nInc=None,
    noise_var=None, scaler=None):

    """
    input:
    - design: a list of design choices that have already been made ['d1'], or ['d1', 'd2'], etc.
    - theta_true: ndarray of true parameter values
    - obs_inds: list of indices where measurements are to be taken
    - calc_qois: scalar, number of qois to calculate
    - qoi_index: list, qois to use in calibration
    - epsInc: scalar, strain increment per load step
    - nInc: scalar, number of strain increments
    - setup_experiment: flag to indicate if strain path
        needs to be generated
    - noise_var: ndarray, experimental noise variance

    output:
    qois from running experiment, which is the
    mps run at the true parameter values

    """

    theta_true = np.atleast_2d(theta_true)
    if scaler is not None:
        theta_true = scaler.inverse_transform(theta_true)

#    boe_step = len(design)

    # join design choices and generate associate strain path
    path = '_'.join(design)
    strain_path = {}
    strain_path[path] =  select_strain_path(\
        epsInc, nInc, path)[path]

    # get string describing strain path (i.e. 'd1_d2')
    nstrain_inc = len(strain_path[path]['strain_11'])
    ob_ind = obs_inds[obs_inds < nstrain_inc]
#    nobs = len(ob_ind)
#    num_obs_per_step = nobs // boe_step

    qois = compute_qois(theta_true, strain_path,
        ob_ind, calc_qois)[path]

    use_qois = qois[:,:,qoi_index]

    # add noise
    if noise_var is not None:
        noisy_qois = norm(use_qois, noise_var**0.5).rvs()
        qois = np.atleast_2d(noisy_qois)
    else:
        qois = use_qois.squeeze()

    return [qois]



def gen_cruciform_tree_data(nsteps, experiment_sample_number, tree_dict,
    build_data_dir, noise_var=None, integrate_displacement=False,
    random_seed=None):


    """
    generate experimental data with set noise to make
    experimental data consistent from trial to trial
    """

    global tree_data_dict
    tree_data_dict = {}

    np.random.seed(40)
    if random_seed is not None:
        np.random.seed(random_seed)

    features = np.load(build_data_dir + "samples.npy")

    for idx, key in enumerate(tree_dict):
        if len(key) == nsteps:
            full_load_path = key
            load_targets = np.load(build_data_dir + f"load_{full_load_path}.npy")
            disp_targets = np.load(build_data_dir + f"disp_{full_load_path}.npy")
            disp = disp_targets[experiment_sample_number, ..., :2, :]
            load = load_targets[experiment_sample_number, ..., :2, :]

            # add noise
            if noise_var[0] is not None:
                disp = norm(disp, noise_var[0]**0.5).rvs()
            disp = np.atleast_2d(disp)
            disp = np.moveaxis(disp, -1, 0)
            if noise_var[1] is not None:
                load = norm(load, noise_var[1]**0.5).rvs()
            load = np.atleast_2d(load)
            load = np.moveaxis(load, -1, 0)
            load = np.expand_dims(load, 1)

            # data integration should not happend here
            # back up indent of lines 188-197 in 'cruciform_experiment'

            # displacement data integration
            if integrate_displacement:
                integrated_disp = [[None for qoi in np.arange(2)] for step in np.arange(nsteps)]
                for step in range(nsteps):
                    for qoi in np.arange(2):
                        disp_T = disp[step, :, qoi].T
                        integrated_disp[step][qoi] = 1/A_roi *\
                            disp_T @ mass_matrix @ disp[step, :, qoi]

                disp = np.expand_dims(np.asarray(integrated_disp), -2)

            tree_data_dict[key] = [disp, load]



def variable_load_step_experiment(design, increment=0.25, noise_var=None, return_pca=False):

    load_path = "".join(design)
    nload_steps = len(load_path)
    data = [None] * 2

    with open(f'experimental_data/variable_increments/increment_{str(increment)}/disp.npy', 'rb') as f:
#    with open(f'experimental_data/cruciform_disp_mismatch_results/disp.npy', 'rb') as f:
        disp_data = np.load(f)
    disp = disp_data[..., :2, :nload_steps]
    # add noise
    if noise_var[0] is not None:
        disp = norm(disp, noise_var[0]**0.5).rvs()
    disp = np.atleast_3d(disp)
    # nsteps x npoints x nqoi
    disp = np.moveaxis(disp, -1, 0)
    if return_pca:
        pca_disp = experimental_data_2_pca(disp, design)

    with open(f'experimental_data/variable_increments/increment_{str(increment)}/load.npy', 'rb') as f:
#    with open(f'experimental_data/cruciform_disp_mismatch_results/load.npy', 'rb') as f:
        load_data = np.load(f)
    load = load_data[:nload_steps, :2]

    if noise_var[1] is not None:
        load = norm(load, noise_var[1]**0.5).rvs()
    load = np.atleast_3d(load)
    load = np.moveaxis(load, -1, 1)
    # nsteps x npoins x nqoi


    if return_pca:
        return [pca_disp[-1, ...], load[-1, ...], disp[-1, ...]]
    else:
        return [disp[-1, ...], load[-1, ...]]



def abbreviated_cruciform_experiment(design, use_objectives=[0,1]):

    load_path = "".join(design)
    with open('simulated_data.pkl', 'rb') as f:
        data = pickle.load(f)
    if 0 in use_objectives and 1 in use_objectives:
        return [data[0][load_path], data[1][load_path], data[2][load_path]]
    elif 0 in use_objectives:
        return [data[0][load_path], data[2][load_path]]
    elif 1 in use_objectives:
        return [data[1][load_path]]



def cruciform_experiment(design, experiment_sample_number,
    qoi_index, tree_dict, build_data_dir, noise_var=None,
    integrate_displacement=False, calc_global_data=False,
    return_pca=False, use_objectives=None,
    theta_true=None):
    """
    input:
    - design: a list of design choices that have already been made ['d1'], or ['d1', 'd2'], etc.
    - experiment_sample_number: sample number from training/test data
          set to be used as experimental data - choose one from test set
    - qoi_index: list, qois to use for each objective in calibration
    - tree_dict: dictionary mapping current load path to a correspoinding
         full load path (5 steps) that starts with the current load path
         e.g.: designs = ['A'], ['A', 'A'], ['A', 'A' ,'A'] and ['A', 'A', 'A', 'A'],
         all correspond to full load path ['A', 'A', 'A', 'A', 'A']
    - build_data_dir: diretory where the training samples live
    - noise_var: ndarray, experimental noise variance for each objective

    output:
    experimental data for each objective, which is a sample/output pair from one of
    the samples generated for testing the surrogates. Optionally:
    with noise added.

    """

    assert use_objectives is not None

    if theta_true is None:
        features = np.load(build_data_dir + "samples.npy")
        theta_true = features[experiment_sample_number, :]

    theta_true = np.atleast_2d(theta_true)

    load_path = "".join(design)
    full_load_path = tree_dict[load_path]
    nload_steps = len(load_path)
    level = nload_steps

    if 0 in use_objectives:
        # nsamples x npoints x nqoi x nsteps
        disp_targets = np.load(build_data_dir + f"disp_{full_load_path}.npy")
        # npoints x nqoi x nsteps
        disp = disp_targets[experiment_sample_number, ..., :2, :level]

        # add noise
        if noise_var[0] is not None:
            disp = norm(disp, noise_var[0]**0.5).rvs()
        disp = np.atleast_2d(disp)
        # nsteps x npoints x nqoi
        disp = np.moveaxis(disp, -1, 0)

        # displacement data integration
        if integrate_displacement:
            integrated_disp = [[None for qoi in np.arange(2)] for step in np.arange(nload_steps)]
            for step in range(nload_steps):
                for qoi in np.arange(2):
                    disp_T = disp[step, :, qoi].T
                    integrated_disp[step][qoi] = 1/A_roi *\
                        disp_T @ mass_matrix @ disp[step, :, qoi]

            int_disp = np.expand_dims(np.asarray(integrated_disp), -2)

        if return_pca:
            pca_disp = experimental_data_2_pca(disp, design)

    if 1 in use_objectives:
        # nsamples x nqoi x nsteps
        load_targets = np.load(build_data_dir + f"load_{full_load_path}.npy")
        # nqoi x nsteps
        load = load_targets[experiment_sample_number, ..., :2, :level]

        if noise_var[1] is not None:
            load = norm(load, noise_var[1]**0.5).rvs()
        load = np.atleast_2d(load)
        load = np.moveaxis(load, -1, 0)
        # nsteps x npoins x nqoi
        load = np.expand_dims(load, 1)


    if len(use_objectives) == 1:
        if 0 in use_objectives:
            if return_pca:
                return [pca_disp[-1, ...], disp[-1, ...]]
            if integrate_displacement:
                return [int_disp[-1, ...], disp[-1, ...]]
            else:
                return [disp[-1, ...]]

        if 1 in use_objectives:
            return [load[-1, ...]]

    elif len(use_objectives) == 2:
        if return_pca:
            return [pca_disp[-1, ...], load[-1, ...], disp[-1, ...]]
        elif integrate_displacement:
            return [int_disp[-1, ...], load[-1, ...], disp[-1, ...]]
        else:
            return [disp[-1, ...], load[-1, ...]]



def cruciform_surr_qois_fun(samples, surr, qoi_index,
    integrate_displacement=False, return_pca=False, use_objectives=None):


    assert use_objectives is not None

    if return_pca and integrate_displacement:
        print("returning pca modes - displacment integration set to 'off' ")
        integrate_displacement = False

    # GP is a list of GP objects, each object belongs
    # to a different node in the load path tree
    disp_surr = surr[0]
    load_surr = surr[1]

    nsteps = len(surr[0])
    samples = np.atleast_2d(samples)
    nsamples = samples.shape[0]

    disp_pred = [None] * nsteps
    load_pred = [None] * nsteps

    pred = [None] * len(use_objectives)

    for step_idx in range(nsteps):

        if 0 in use_objectives:
            gp_disp_pred = \
                disp_surr[step_idx].value(samples, return_pca=return_pca)
            disp_pred[step_idx] = gp_disp_pred[..., qoi_index[0]]

        if 1 in use_objectives:
            gp_load_pred = load_surr[step_idx].value(samples)
            load_pred[step_idx] = gp_load_pred[..., qoi_index[1]]


    if 0 in use_objectives:
        # dimensions: nsteps x nsamples x nobs x nqoi
        disp_pred = np.atleast_2d(np.asarray(disp_pred))

        # displacement data integration
        if integrate_displacement:
            integrated_disp = [[None for qoi in np.arange(2)] for step in np.arange(nsteps)]
            for step in range(nsteps):
                for qoi in np.arange(2):
                    # create an nsamples x nobs x 1 array
                    disp = np.expand_dims(disp_pred[step, ..., qoi], -1)
                    # transpose the last two dimensions: nsamples x 1 x nobs
                    disp_T = np.moveaxis(disp.T, -1, 0)
                    integrated_disp[step][qoi] = 1/A_roi *\
                        np.matmul(disp_T,\
                        np.matmul(mass_matrix, disp)).squeeze(axis=-1)
            disp_pred = np.moveaxis(np.asarray(integrated_disp), 1, -1)
        if nsamples == 1:
            disp_pred = disp_pred.squeeze(axis=1)

    if 1 in use_objectives:
        # dimensions: nsteps x nsamples x nobs x nqoi
        load_pred = np.atleast_2d(np.asarray(load_pred))
        if nsamples == 1:
            load_pred = load_pred.squeeze(axis=1)

    if len(use_objectives) == 1:
        if 0 in use_objectives:
            pred[0] = disp_pred
        elif 1 in use_objectives:
            pred[0] = load_pred
    elif len(use_objectives) == 2:
        pred = [disp_pred, load_pred]
#    return [disp_pred[..., ::350, :], load_pred]
    return pred



def mps_qois_fun(samples, design,
    obs_inds, num_qois, qoi_index, epsInc,
    nInc, return_steps='all', scaler=None,
    return_strain_path=False):
    """
    Input:
    -design:
    """
    # compute_qois accepts a dictionary 'strain_paths'
    # with key-value pair being the
    # path name and associate strain
    # and returns stresses.

    samples = np.atleast_2d(samples)
    if scaler is not None:
        samples = scaler.inverse_transform(samples)

    boe_step = len(design)
    # get string describing strain path (i.e. 'd1_d2')
    path = '_'.join(design)
    strain_path = {}
    strain_path[path] = select_strain_path(\
        epsInc, nInc, path)[path]

    # number of total simulation steps so far (nInc * nsteps) - (nsteps - 1)
    nstrain_inc = len(strain_path[path]['strain_11'])
    ob_ind = obs_inds[obs_inds < nstrain_inc]
    nobs = len(ob_ind)
    num_obs_per_step = nobs // boe_step

    qois = compute_qois(samples, strain_path,
        ob_ind, num_qois)[path]

    if return_steps == 'all':
       use_qois = qois[:,:,qoi_index]
    elif return_steps == 'last':
       use_qois = qois[:,-num_obs_per_step:, qoi_index]

    if return_strain_path:
        return use_qois, strain_path
    else:
        return [use_qois]



def GP_qois_fun_orig(samples, design, surrogates,
    input_scaler, scale=True,
    return_std=False, return_steps='all',
    data_scaler=None, scale_data=False):

    # GP is a list of GP dictionaries, each dictionary belongs
    # to a different leg of the path and contains a separate
    # GP for each qoi and each obsevation

    samples = np.atleast_2d(samples)
    if return_steps == 'all':
        GPs = [None] * len(design)
        for p in range(len(design)):
            path = '_'.join(design[:p+1])
            GPs[p] = surrogates[path]
    elif return_steps == 'last':
        GPs = [None] * 1
        path = '_'.join(design)
        GPs[0] = surrogates[path]

    nsamples = samples.shape[0]
    key_list = list(GPs[0].keys())
    nqois = len([i for i in key_list if 'qoi_' in i])
    nobs = len(GPs[0][key_list[0]])
    qois = np.zeros((nsamples, len(GPs)*nobs, nqois))

    idx = -1
    for gp in range(len(GPs)): #loop through dictionaries (one per included design step)
        for ob in range(nobs):
            idx += 1
            for qoi in range(nqois):
                ob_GP = GPs[gp]['qoi_' + str(qoi)]['observation_' + str(ob)]
                qois[:, idx, qoi] = eval_GP(\
                    samples,
                    ob_GP['GP'],
                    input_scaler,
                    return_std=return_std,
                    scale=scale)
    use_qois = qois
    if scale_data:
        use_qois_scaled = data_scaler.transform(\
            use_qois.reshape(-1, nqois))
        return use_qois_scaled.reshape(use_qois.shape)
    else:
        return [use_qois]



def GP_qois_fun(samples, design, surrogates,
    input_scaler, obs, scale=True,
    return_std=False, return_steps='all',
    data_scaler=None, scale_data=False):

    # GP is a list of GP dictionaries, each dictionary belongs
    # to a different leg of the path and contains a separate
    # GP for each qoi and each obsevation
    boe_step = len(design)
    nobs_pstep = len(obs)

    samples = np.atleast_2d(samples)
    if return_steps == 'all':
        GPs = [None] * boe_step
        for p in range(len(design)):
            path = '_'.join(design[:p+1])
            GPs[p] = surrogates[path]
    elif return_steps == 'last':
        GPs = [None] * 1
        path = '_'.join(design)
        GPs[0] = surrogates[path]

    nobs = nobs_pstep * len(GPs) # total number of observations to be returned per qoi
    nsamples = samples.shape[0]
    key_list = list(GPs[0].keys())
    nqois = len([i for i in key_list if 'qoi_' in i])
    qois = np.zeros((nsamples, len(GPs), nobs_pstep, nqois))

    for gp in range(len(GPs)): #loop through dictionaries (one per included design step)
        for ob in range(nobs_pstep): #loop through qois
            for qoi in range(nqois):
                qoi_GP = GPs[gp]['qoi_' + str(qoi)]['observation_' + str(ob)]
                qois[:, gp, ob, qoi] = eval_GP(\
                    samples,
                    qoi_GP['GP'],
                    input_scaler,
                    return_std=return_std,
                    scale=scale)
    use_qois = qois.reshape(nsamples, -1, nqois)

    if scale_data:
        use_qois_scaled = data_scaler.transform(use_qois)
        return use_qois_scaled
    else:
        return [use_qois]


def displacement_integration(disp_data, nqoi=2):

    integrated_disp = [None] * nqoi
    for qoi in np.arange(nqoi):
        # create an nsamples x nobs x 1 array
        disp = np.expand_dims(disp_data[..., qoi], -1)
        # transpose the last two dimensions: nsamples x 1 x nobs
        disp_T = np.moveaxis(disp.T, -1, 0)
        integrated_disp[qoi] = 1/A_roi *\
            np.matmul(disp_T,\
            np.matmul(mass_matrix, disp)).squeeze(axis=-1)
    integrated_disp = np.moveaxis(np.asarray(integrated_disp), 0, -1)
    return integrated_disp



def gen_mass_matrix():
    from exodus_file_setup import get_node_idx, setup_exodus_files
    global mass_matrix
    base_mesh = "cruciform_surrogates/surrogate_surface.e"
    node_idx = get_node_idx(base_mesh)
    mass_matrix_unordered = np.load('cruciform_surrogates/calibration_data/cruciform_roi_mass_matrix.npy')
    mass_matrix = mass_matrix_unordered[node_idx, :]
    global A_roi
    A_roi = np.sum(mass_matrix)



def load_cruciform_surrogate(design, surrogate_type, return_steps="all", nmodes=5,
    surrogate_dir=None):

#    surrogate_dir = f"cruciform_surrogates/cruciform_{surrogate_type}"

    boe_step = len(design)
    load_path = "".join(design)
    if return_steps == 'all':
        nsteps = boe_step
        disp_surr = [None] * nsteps
        load_surr = [None] * nsteps
        for p in range(len(design)):
            path = ''.join(design[:p+1])
            with open(f"{surrogate_dir}/disp_{surrogate_type}_{nmodes}_modes_{path}.pkl", 'rb') as f:
                disp_surr[p] = pickle.load(f)
            with open(f"{surrogate_dir}/load_{surrogate_type}_{path}.pkl", 'rb') as f:
                load_surr[p] = pickle.load(f)

    elif return_steps == 'last':
        nsteps = 1
        path = ''.join(design)
        with open(f"{surrogate_dir}/disp_{surrogate_type}_{nmodes}_modes_{path}.pkl", 'rb') as f:
            disp_surr = [pickle.load(f)]
        with open(f"{surrogate_dir}/load_{surrogate_type}_{path}.pkl", 'rb') as f:
            load_surr = [pickle.load(f)]

    return [disp_surr, load_surr]



def gen_pca_dict(tree_dict, surrogate_dir, surr_type, nmodes):

    global pca_dict
    pca_dict = {}

    for p, path in enumerate(tree_dict):
        try:
            with open(f"{surrogate_dir}/disp_{surr_type}_{nmodes}_modes_{path}.pkl", 'rb') as f:
                surr = pickle.load(f)
        except:
            print(f"problem with path {path}")
            continue
        pca_dict[path] = {}

        if surr.output_scaling:
            output_scaler = [surr.output_scaler[i] for i in np.arange(3)]
            pca_dict[path]['output_scaler'] = output_scaler
        else:
            pca_dict[path]['output_scaler'] = None

        pca_mean = [surr.pca[i].mean_ for i in np.arange(3)]
        pca_dict[path]['pca_mean'] = pca_mean

        pca_basis = [surr.pca[i].components_ for i in np.arange(3)]
        pca_dict[path]['pca_basis'] = pca_basis
        pca_dict[path]['pca'] = surr.pca


def pca_noise_var(load_path, noise_var, nqoi=3, return_steps='all'):


    # V is orthonormal, so V^T = V^-1
    # multiplying by VV^T should yield the identity matrix if it is orthonormal

    nsteps = len(load_path)
    pca_var = [[None for q in np.arange(nqoi)] for ss in np.arange(nsteps)]

    for ss in np.arange(nsteps):
        path = ''.join(load_path[:ss+1])

        output_scaler = pca_dict[path]['output_scaler']
        bases = pca_dict[path]['pca_basis']

        for qoi in np.arange(nqoi):
            V = bases[qoi]
            N = V.shape[1]
            ncomponents = V.shape[0]
            Sigma = np.eye(N) * noise_var
            if output_scaler is not None:
                scale = output_scaler[qoi].scale_
                pca_var[ss][qoi] = np.dot(V, np.dot((1/scale**2)*Sigma, V.T))
            else:
#                pca_var[ss][qoi] = np.dot(V, np.dot(Sigma, V.T))
                pca_var[ss][qoi] = np.eye(ncomponents) * noise_var
    if return_steps == 'last':
        return np.expand_dims(np.asarray(pca_var[-1]), 0)
    elif return_steps == 'all':
        return np.asarray(pca_var)



def experimental_data_2_pca(data, design, reverse=False):


    """
    data is an nsteps x nsamples x nobs x nqoi array
    or an nsteps x nobs x nqoi array
    """

    squeeze_data = False
    if np.ndim(data) == 3:
        data = np.expand_dims(data, 1)
        squeeze_data = True
    elif np.ndim(data) > 4:
        raise ValueError('Check data dimensions for pca transformation')
    elif np.ndim(data) < 3:
        raise ValueError('Check data dimensions for pca transformation')

    nsteps = data.shape[0]
    nqois = data.shape[-1]
    nsamples = data.shape[1]

    for key, item in pca_dict.items():
        first_key = key
        break
    ncomponents = pca_dict[first_key]['pca_basis'][0].shape[0]

    if not reverse:
        pca_data = np.zeros((nsteps, nsamples, ncomponents, nqois))

        for ss in np.arange(nsteps):

            if nsteps == len(design):
                path = ''.join(design[:ss+1])
            elif nsteps == 1:
                path = ''.join(design)

            pca_info = pca_dict[path]

            for qoi in np.arange(nqois):

                pca_mean = pca_info['pca_mean'][qoi]
                pca_basis = pca_info['pca_basis'][qoi]

                qoi_data = data[ss, ..., qoi]
                if pca_info['output_scaler'] is not None:
                    output_scaler = pca_info['output_scaler'][qoi]
                    scaled_data = output_scaler.transform(qoi_data)
                else:
                    scaled_data = qoi_data
                pca_data[ss, ..., qoi] = (scaled_data - pca_mean) @ pca_basis.T

                if False:
                    import matplotlib.pyplot as plt
                    import matplotlib
                    fig, axs = plt.subplots(1, 2)
                    axs[0].plot(qoi_data, 'r.', label='disp')
                    axs[1].plot(scaled_data.T, 'b.', label='0 mean unit variance')
                    plt.suptitle(f"qoi: {qoi}, step: {ss}")
                    axs[0].legend()
                    axs[1].legend()
                    plt.show()

        if squeeze_data:
            return pca_data.squeeze(axis=1)
        else:
            return pca_data

    elif reverse:

        npts = pca_dict['A']['pca_basis'][0].shape[1]
        field_data = np.zeros((nsteps, nsamples, npts, nqois))

        for ss in np.arange(nsteps):

            if nsteps == len(design):
                path = ''.join(design[:ss+1])
            elif nsteps == 1:
                path = ''.join(design)

            pca_info = pca_dict[path]

            for qoi in np.arange(nqois):

                pca = pca_info['pca'][qoi]
                qoi_data = data[ss, ..., qoi]
                scaled_field_data = pca.inverse_transform(qoi_data)
                if pca_info['output_scaler'] is not None:
                    output_scaler = pca_info['output_scaler'][qoi]
                    field_data[ss, ..., qoi] = output_scaler.inverse_transform(scaled_field_data)
                else:
                    field_data[ss, ..., qoi] = scaled_field_data

        return field_data


def MTS_standin(total_load_path, return_pca=True, use_objectives=[0,1]):

    path_str = "".join(total_load_path)
    data_path = "../data/ST10_test_data_07_16"
    disp_data = np.loadtxt(f"{data_path}/averaged_disp_data_{path_str}.txt")
    load_data = np.loadtxt(f"{data_path}/averaged_load_data_{path_str}.txt")

    if return_pca:
        disp_data = experimental_data_2_pca(np.expand_dims(disp_data, 0),
            total_load_path).squeeze()

    if 0 in use_objectives and 1 in use_objectives:
        return [disp_data, load_data]
    elif 0 in use_objectives:
        return [disp_data]
    elif 1 in use_objectives:
        return [load_data]

