import numpy as np

def polyvert_pca(polygon):
    u = np.mean(polygon, axis=0)
    cov_mat = np.cov(polygon - u, rowvar=False)
    assert not np.isnan(cov_mat).any() and not np.isinf(cov_mat).any() and check_symmetric(cov_mat)
    eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
    sort_indices = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, sort_indices]
    pc_lengths = eig_vals[sort_indices]
    return eig_vecs, pc_lengths, u

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def is_scallop(pca_lengths_sorted):
    w_to_l = pca_lengths_sorted[0] / pca_lengths_sorted[1]
    scallop_plane_h_d = pca_lengths_sorted[2]
    return w_to_l > 1.0 and w_to_l < 3 and scallop_plane_h_d < 0.05

def remove_outliers(pnts, radius):
    mean_distances = np.linalg.norm(pnts - np.mean(pnts, axis=1)[:, None], axis=0)
    inlier_mask = mean_distances < radius
    return pnts[:, inlier_mask]
