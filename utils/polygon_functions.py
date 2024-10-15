import numpy as np
import cv2
from tqdm import tqdm
from shapely.geometry import *
import re
import math
from utils import geo_utils

RNN_DISTANCE = 0.030
SCALLOP_MAX_SIZE = 0.15

CLUSTER_CNT_THRESH = 2
CENTROID_ERR_THRESH = 0.05  # Threshold distance of centroid to cluster centroid to be inlier

CNN_CONF_THRESH = 0.5

def filter_polygon_detections(polygon_detections):
    # Filter on polygon contour shape, CNN confidence, symmetry in width dimension, convexity, curve, pca statistics
    valid_polygons = []
    invalid_polygons = []
    for polygon, conf in polygon_detections:
        eig_vecs, eig_vals, center_pnt = pca(polygon)
        if is_scallop_eigs(eig_vals) and conf > CNN_CONF_THRESH:
            valid_polygons.append(polygon)
        else:
            invalid_polygons.append(polygon)
    return np.array(valid_polygons, dtype=object), np.array(invalid_polygons, dtype=object)

FOV_MUL = 1.2

def pnt_in_cam_fov(pnt, cam_fov, tol_deg=0):
    tol_rad = math.radians(tol_deg)
    vec_zy = pnt * np.array([0, 1, 1])
    vec_zy /= np.linalg.norm(vec_zy) + 1e-9
    vec_zx = pnt * np.array([1, 0, 1])
    vec_zx /= np.linalg.norm(vec_zx) + 1e-9
    angle_x = math.acos(vec_zx[2])
    angle_y = math.acos(vec_zy[2])
    angle_thresh = FOV_MUL * cam_fov / 2
    return angle_x < angle_thresh[0] + tol_rad and angle_y < angle_thresh[1] + tol_rad

def in_camera_fov(polygon_cam, cam_fov, tol_deg=10):
    polygon_center = np.mean(polygon_cam, axis=1)
    return pnt_in_cam_fov(polygon_center, cam_fov, tol=tol_deg)

def is_scallop_eigs(pca_eigen_values):
    w_to_l = pca_eigen_values[0] / pca_eigen_values[1]
    scallop_plane_h_d = pca_eigen_values[2]
    return w_to_l > 1.0 and w_to_l < 3 and scallop_plane_h_d < 0.110

def filter_clusters(clusters):
    # Filter on cluster cnt, polygon area consistency
    valid_clusters = []
    invalid_clusters = []
    for cluster in clusters:
        # filter polygons inside cluster
        # polygon_centroids = [np.mean(poly, axis=0) for poly in cluster]
        # cluster_centroid = np.mean(polygon_centroids, axis=0)
        # cluster_width = calc_cluster_widths([cluster], mode='max')
        # centroid_dist = np.linalg.norm(polygon_centroids - cluster_centroid, axis=1)
        # inlier_cluster =
        # print(cluster_centroid)
        # CENTROID_ERR_THRESH
        if len(cluster) >= CLUSTER_CNT_THRESH:
            valid_clusters.append(cluster)
        else:
            invalid_clusters.append(cluster)
    return valid_clusters, invalid_clusters

PC_MUL = 1.9
def polygon_PCA_width(polygon):
    pc_vecs, pc_lengths, center_pnt = pca(polygon)
    pc_lengths = np.sqrt(pc_lengths) * PC_MUL
    scaled_pc_lengths = pc_lengths * 2
    return scaled_pc_lengths[0]

def polygon_max_width(polygon):
    scallop_vert_mat = np.repeat(polygon[None], polygon.shape[0], axis=0)
    scallop_vert_dists = np.linalg.norm(scallop_vert_mat - scallop_vert_mat.transpose([1, 0, 2]), axis=2)
    return np.max(scallop_vert_dists)

def calc_cluster_widths(polygon_clusters, mode=None):
    cluster_widths_l = []
    if mode == 'max':
        sz_func = polygon_max_width
    elif mode == 'pca':
        sz_func = polygon_PCA_width
    else:
        sz_func = lambda poly: (polygon_max_width(poly) + polygon_PCA_width(poly)) / 2
    for cluster in tqdm(polygon_clusters):
        cluster_poly_width = np.mean([sz_func(poly) for poly in cluster])
        cluster_poly_width = min(float(cluster_poly_width), SCALLOP_MAX_SIZE)
        cluster_widths_l.append(cluster_poly_width)
    return cluster_widths_l

def get_next_seed_index(mask_arr):
    for i, val in enumerate(mask_arr):
        if val == True:
            return i

def rnn_clustering(point_groups):
    unclustered_mask = np.ones((len(point_groups),)).astype(np.bool)
    neighbourhood_mask = unclustered_mask.copy()
    centers = np.array([np.mean(poly, axis=0) for poly in point_groups])
    cluster_indexes = []
    while any(unclustered_mask):
        seed_center = centers[get_next_seed_index(unclustered_mask)]
        for i in range(2):
            unclst_dists = np.linalg.norm(centers - seed_center, axis=1)
            neighbourhood_mask = (unclst_dists < RNN_DISTANCE) * unclustered_mask
            seed_center = np.mean(centers[neighbourhood_mask], axis=0)
        neighbour_idxs = np.where(neighbourhood_mask)[0]
        cluster_indexes.append(neighbour_idxs)
        unclustered_mask[neighbour_idxs] = False
    return cluster_indexes

def polygon_rnn_clustering(polygons, labels):
    cluster_idxs = rnn_clustering(polygons)
    polygon_clusters = []
    clustered_labels = []
    for neighbour_idxs in cluster_idxs:
        polygon_clusters.append([polygons[idx] for idx in neighbour_idxs])
        clustered_labels.append([labels[idx] for idx in neighbour_idxs])
    return polygon_clusters, clustered_labels

def UpsamplePoly(polygon, num=10):
    poly_ext = np.append(polygon, [polygon[0, :]], axis=0)
    up_poly = []
    for idx in range(poly_ext.shape[0]-1):
        int_points = np.linspace(poly_ext[idx], poly_ext[idx+1], num=num)
        up_poly.extend(int_points)
    return np.array(up_poly)

def Project2Img(points, cam_mtx, dist):
    result = []
    rvec = tvec = np.array([0.0, 0.0, 0.0])
    if len(points) > 0:
        result, _ = cv2.projectPoints(points, rvec, tvec,
                                      cam_mtx, dist)
    return np.squeeze(result, axis=1)

def undistort_pixels(pixels, cam_mtx, dist):
    pix_ud = np.array([])
    if len(pixels) > 0:
        pix_ud = cv2.undistortPoints(np.expand_dims(pixels.astype(np.float32), axis=1), cam_mtx, dist, P=cam_mtx)
    return np.squeeze(pix_ud, axis=1)

def pca(points):
    u = np.mean(points, axis=0)
    cov_mat = np.cov(points - u, rowvar=False)
    assert not np.isnan(cov_mat).any() and not np.isinf(cov_mat).any() and check_symmetric(cov_mat)
    eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
    sort_indices = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, sort_indices]
    pc_lengths = eig_vals[sort_indices]
    return eig_vecs, pc_lengths, u

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def remove_outliers(pnts, radius):
    mean_distances = np.linalg.norm(pnts - np.mean(pnts, axis=1)[:, None], axis=0)
    inlier_mask = mean_distances < radius
    return pnts[:, inlier_mask]

def polyline_dist_thresh(pnt, polyline, thresh):
    for seg_idx in range(len(polyline)-1):
        line_seg = [polyline[seg_idx+1], polyline[seg_idx]]
        dist = pnt2lineseg_dist(pnt, line_seg)
        if dist < thresh:
            return True
    return False

def pnt2lineseg_dist(point, line):
    # unit vector
    unit_line = line[1] - line[0]
    norm_unit_line = unit_line / np.linalg.norm(unit_line)

    # compute the perpendicular distance to the theoretical infinite line
    segment_dist = (
            np.linalg.norm(np.cross(line[1] - line[0], line[0] - point)) /
            np.linalg.norm(unit_line)
    )

    diff = (
            (norm_unit_line[0] * (point[0] - line[0][0])) +
            (norm_unit_line[1] * (point[1] - line[0][1]))
    )

    x_seg = (norm_unit_line[0] * diff) + line[0][0]
    y_seg = (norm_unit_line[1] * diff) + line[0][1]

    endpoint_dist = min(
        np.linalg.norm(line[0] - point),
        np.linalg.norm(line[1] - point)
    )

    # decide if the intersection point falls on the line segment
    lp1_x = line[0][0]  # line point 1 x
    lp1_y = line[0][1]  # line point 1 y
    lp2_x = line[1][0]  # line point 2 x
    lp2_y = line[1][1]  # line point 2 y
    is_betw_x = lp1_x <= x_seg <= lp2_x or lp2_x <= x_seg <= lp1_x
    is_betw_y = lp1_y <= y_seg <= lp2_y or lp2_y <= y_seg <= lp1_y
    if is_betw_x and is_betw_y:
        return segment_dist
    else:
        return 10.0