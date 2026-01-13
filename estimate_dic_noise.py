import numpy as np
import os, sys
import pickle
from postprocess import register_and_remap_disp_data, get_dic_center_circle, knn_remap, extract_load_data
from scipy.io import loadmat
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib
from exodus_file_setup import get_node_idx, setup_exodus_files
from sklearn.decomposition import PCA


def separate_signal_and_noise(Y, rank=None, explained_variance=0.95):
    """
    Separate the singal form the noise in given field data.

    Paramters:
    ----------
    Y : ndarray of shape (M, N)
        M repeated measurements (each of size N)
    rank: int or None, optional
        * rank = 0 --> treat the mean field as signal (classic mean-only removal)
        * rank = k>0 --> remove the first k PCA modes of the centered data
        * rank = None (default) --> keep just enough PCA modes to explain 'explained_variance' of total variance
    explained_variance : float, default 0.95
        Cumulative variance fraction to retain when 'rank' is None.

    Returns:
    -------
    Y_signal : ndarray of shape (1, N)
      Estimated measurement signal
    Y_noise : ndarray of shape (M, N)
      Estimated measurement noise
    k_used : int
      Number of principal components used for the signal model (0 means mean-field only)
    """

    M, N = Y.shape

    # Compute mean and center data
    Y_mean = Y.mean(axis=0)
    Y_centered = Y - Y_mean

    # Case 1: mean-only signal removal
    if rank == 0:
        Y_signal = Y_mean
        k_used = 0

    # Case 2: PCA with user-specified number of modes
    elif isinstance(rank, int):
        k_used = rank if rank is not None else 0 # safeguard
        if k_used > 0:
            pca = PCA(n_components=k_used)
            coeff = pca.fit_transform(Y_centered)
            Y_signal_centered = pca.inverse_transform(coeff)
            Y_signal = Y_signal_centered + Y_mean
        else:
            Y_signal = Y_mean

    # Case 3: Automatic PCA based on explained variance
    else: # rank is None
        pca_full = PCA().fit(Y_centered)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        k_used = int(np.searchsorted(cumvar, explained_variance) + 1)
        pca = PCA(n_components=k_used)
        coeff = pca.fit_transform(Y_centered)
        Y_signal_centered = pca.inverse_transform(coeff)
        Y_signal = Y_signal_centered + Y_mean

    # Noise residuals
    Y_noise = Y - Y_signal

    return Y_signal, Y_noise, k_used


def estimate_noise_covariance(Y, rank=None, explained_variance=0.95):
    """
    Estimate the measurement noise covariance by removing signal structure.

    This isolates the noise component by subtracting either the mean field (rank=None)
    or a low-rank PCA reconstruction (rank=k) from the repeated measurements before
    computing the covariance.

    Paramters:
    ----------
    Y : ndarray of shape (M, N)
        M repeated measurements (each of size N)
    rank: int or None, optional
        * rank = 0 --> treat the mean field as signal (classic mean-only removal)
        * rank = k>0 --> remove the first k PCA modes of the centered data
        * rank = None (default) --> keep just enough PCA modes to explain 'explained_variance' of total variance
    explained_variance : float, default 0.95
        Cumulative variance fraction to retain when 'rank' is None.

    Returns:
    -------
    Sigma_noise : ndarray of shape (N, N)
      Estimated measurement noise covariance (signal removed)
    k_used : int
      Number of principal components used for the signal model (0 means mean-field only)
    """

    M, N = Y.shape
    Y_signal, Y_noise, k_used = separate_signal_and_noise(Y, rank=rank, explained_variance=explained_variance)
    Sigma_single = (Y_noise.T @ Y_noise) / (M - 1) # for a single repeat
    Sigma_mean = Sigma_single / M # Covariance of the mean field

    return Sigma_mean, k_used


def estimate_load_noise_var(Y, num_frames_to_average, rank=None):
    X_avg_load = Y[-num_frames_to_average:, :2].mean(axis=1)
    Y_avg_load = Y[-num_frames_to_average:, -2:].mean(axis=1)
    Y_avg = np.array([X_avg_load, Y_avg_load]).T
    Y_signal, Y_noise, _ = separate_signal_and_noise(Y_avg, rank=rank)
    noise_var = np.diag( (Y_noise.T @ Y_noise) / (num_frames_to_average - 1) )
    return 1/num_frames_to_average * noise_var

def compute_distance_matrix(coords):
    """ Compute the pairwise distance matrix for spatial coordinates.

    Paramters:
    ----------
    coords : ndarray of shape (N, 2)
        Coordinates of the N spatial points (x, y)

    Returns:
    --------
    D : ndarray of shape (N, 2)
      Pairwise euclidean distances
    """

    return cdist(coords, coords, metric='euclidean') # Pairwise distances


def plot_correlation_vs_distances(Sigma, coords, i, file_id):
    """
    Plot the correlation vs. distance to asses spatial structure.

    Paramters:
    ----------
    Sigma : ndarray of shape (N, N)
        Empirical covariane matrix
    coords : ndarray of shape (N, 2)
        Spatial coordinates of measurement points
    """

    D = compute_distance_matrix(coords)
    N = Sigma.shape[0]
    i_upper = np.triu_indices(N, k=1)
    var_i = np.diag(Sigma)[i_upper[0]]
    var_j = np.diag(Sigma)[i_upper[1]]
    denom = np.sqrt(var_i * var_j)
    corrs = Sigma[i_upper] / denom
    dists = D[i_upper]

    plt.scatter(dists, corrs, alpha=0.3)
    plt.xlabel("Distance between points")
    plt.ylabel("Empirical Correlation")
    plt.title("Correlation vs. Distance (Spatial Structure Check)")
    plt.grid(True)
    plt.savefig(f"/home/dericci/mc2/data/noise_approximation/measurement_noise_spatial_structure_{i}_{file_id}.png")
    plt.close()

    corr_matrix = np.corrcoef(Sigma)
    plt.figure()
    plt.imshow(corr_matrix, aspect='auto')
    plt.colorbar()
    plt.title('Correlation matrix')
    plt.xlabel('Spatial point index')
    plt.ylabel('Spatial point index')
    plt.savefig(f"/home/dericci/mc2/data/noise_approximation/correlation_mat_{i}_{file_id}.png")
    plt.close()


def plot_covariance_matrix(Sigma, i, file_id):
    """ Plot the measurement noise covariance matrix

    Parameters:
    ----------
    Sigma : ndarray of shape (N, N)
    """

    plt.plot()
    plt.imshow(Sigma)
    plt.colorbar()
    plt.savefig(f"/home/dericci/mc2/data/noise_approximation/measurement_noise_cov_{i}_{file_id}.png")
    plt.close()


def create_noise_exodus_file(Y_mean, Sigma, file_id, rank=None, explained_variance=0.95):

    disp_conv_factor = 10**6 # (m) to (um)
    base_mesh = '/home/dericci/mc2/code/surrogate_surface.e'
    node_idx = get_node_idx(base_mesh)
    dirname = '/home/dericci/mc2/data/noise_approximation'
    exodus_filename = f"{dirname}/noise_approx_{file_id}.e"
    field_names = ["node_index", "ux_mean_field", "uy_mean_field",
                                 "ux_noise", "uy_noise"]
    global_var_names = None
    output_database, surr_node_idx =\
        setup_exodus_files(base_mesh, exodus_filename, field_names,
        global_var_names)

    ncoords = Y_mean.shape[0]
    exo_output_times = np.arange(1) + 1
    output_times = np.arange(1, dtype=float)

    str_comps = ["x", "y"]
    for file in range(1):
        output_database.put_time(exo_output_times[0], output_times[0])
        for ii, str_comp in enumerate(str_comps):
            noise_std = np.diag(Sigma[..., ii]) ** 0.5
            # displacement nodal values
            output_database.put_node_variable_values("node_index", 1,
                surr_node_idx)
            output_database.put_node_variable_values("u" + str_comp + "_mean_field", 1,
                Y_mean[surr_node_idx, ii]*disp_conv_factor)
            output_database.put_node_variable_values("u" + str_comp + "_noise",  1,
                noise_std[surr_node_idx]*disp_conv_factor)

    output_database.close()


def estimate_noise(data_folder, load_path, rank=None, num_frames_to_average=48):

    remap = False
    if rank == 0:
        file_id = 'mean_signal'
    elif isinstance(rank, int):
        file_id = f'PCA_rank_{rank}'
    else:
        file_id = 'automatic_PCA'

    if remap:
        target_radius = 20.3
        m_to_mm_scale = 1e3
        num_neighbors=9

        dic_coords, dic_fields, target_coords = register_and_remap_disp_data(data_folder, 0)
        nfiles = dic_fields.shape[-1]
        ncoords = target_coords.shape[0]

        disp_fields = np.zeros((ncoords, 2, nfiles))
        for file in range(nfiles):
            print(f"Remapping file {file}")
            dic_center_coords, dic_center_fields = get_dic_center_circle(
                dic_coords, dic_fields[..., file], target_radius) 

            # disp reported in (m) to compare with surrogate output
            disp_fields[..., file] = knn_remap(dic_center_coords[:, :2],
                dic_center_fields, target_coords,
                num_neighbors=num_neighbors) / m_to_mm_scale

        remapped = {'fields': disp_fields, 'coords': target_coords}
        with open(f'{data_folder}/remapped_data.pkl', 'wb') as f:
            pickle.dump(remapped, f)
    else:
        with open(f'{data_folder}/remapped_data.pkl', 'rb') as f:
            remapped = pickle.load(f)
        disp_fields = remapped['fields']
        target_coords = remapped['coords']

    ncoords = target_coords.shape[0]
    Sigma_empirical = np.zeros((ncoords, ncoords, 2))
    for i in range(2):
        Sigma_empirical[..., i], k_used = estimate_noise_covariance(disp_fields[:, i, :].T, rank=rank) # (m)
        plot_correlation_vs_distances(Sigma_empirical[..., i], target_coords, i, file_id)
        plot_covariance_matrix(Sigma_empirical[..., i], i, file_id)

    with open(f'{data_folder}/disp_noise_cov_{file_id}.npy', 'wb') as f:
        np.save(f, Sigma_empirical)

    # Load reported in (kN) in raw data
    # Convert to (N) for calibration against surrogates
    load_filename = data_folder + f"load-step-{load_path}2.csv"
    load_data = np.atleast_2d(extract_load_data(load_filename, 0)) * 1000 # (N)
    load_noise_var = estimate_load_noise_var(load_data, 48, rank=rank) # (N)

    with open(f'{data_folder}/load_noise_var_{file_id}.npy', 'wb') as f:
        np.save(f, load_noise_var)

    mean_field = register_and_remap_disp_data(data_folder, 48) # (m)
    create_noise_exodus_file(mean_field,  Sigma_empirical, file_id, rank=rank)

    return Sigma_empirical, load_noise_var

if __name__ == '__main__':
    remap = False
    rank = 0
    load_path = 'A'
    data_folder = f'/home/dericci/mc2/data/ST10_test_data_07_16/load-step-{load_path}/'
    disp_noise_cov, load_noise_var = estimate_noise(data_folder, load_path, rank=rank)
