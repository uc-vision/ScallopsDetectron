import numpy as np
import cv2
from tqdm import tqdm
import Metashape
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
def in_camera_fov(polygon_cam, cam_fov):
    polygon_center = np.mean(polygon_cam, axis=1)
    vec_zy = polygon_center * np.array([0, 1, 1])
    vec_zy /= np.linalg.norm(vec_zy) + 1e-9
    vec_zx = polygon_center * np.array([1, 0, 1])
    vec_zx /= np.linalg.norm(vec_zx) + 1e-9
    angle_x = math.acos(vec_zx[2])
    angle_y = math.acos(vec_zy[2])
    angle_thresh = FOV_MUL * cam_fov / 2
    return angle_x < angle_thresh[0] and angle_y < angle_thresh[1]

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

    # K-means clustering with class reduction if centers too close
    # centroids = points if k_means == -1 else points[np.random.randint(points.shape[0], size=k_means)]
    # k_means = centroids.shape[0]
    # classes = np.zeros(points.shape[0], dtype=np.float64)
    # distances = np.zeros([points.shape[0], k_means], dtype=np.float64)
    # for itt in tqdm(range(K_MEANS_MAXITTS)):
    #     for cnt_idx, c in enumerate(centroids):
    #         distances[:, cnt_idx] = np.linalg.norm(points - c, axis=1)
    #     classes = np.argmin(distances, axis=1)
    #     c = 0
    #     while (c < centroids.shape[0]):
    #         centroids[c] = np.mean(points[classes == c], 0)
    #         centroid_dists = np.concatenate([centroids[:c], centroids[c+1:]], axis=0) - centroids[c]
    #         if centroid_dists.shape[0] > 0:
    #             if np.min(np.linalg.norm(centroid_dists, axis=1)) < CENTROID_MERGE_DIST:
    #                 centroids = np.delete(centroids, c, axis=0)
    #                 break
    #         c += 1
    #
    # cluster_counts = np.array([len(points[classes == c]) for c in range(len(centroids))])
    # return centroids, cluster_counts, classes


def get_chunk_polygons_dict(chunk, key=None, world_crs=False):
    chunk_marker_dict = {v.key: v.position for v in chunk.markers}
    chunk_elavation = chunk.elevation
    chunk_transform = chunk.transform.matrix
    T_inv = chunk_transform.inv()
    shapes_crs = chunk.shapes.crs
    assert chunk.marker_crs is None

    chunk_polygons = {}
    for shape in tqdm(chunk.shapes.shapes):
        group_label = shape.group.label
        shape_label = shape.label
        if key is not None and key not in group_label:
            continue
        if shape.geometry.type != Metashape.Geometry.Type.PolygonType:
            continue
        vertex_coords = shape.geometry.coordinates[0]
        if isinstance(vertex_coords[0], int):
            # Markers
            # print("Markers")
            scallop_poly = [chunk_marker_dict[idx] for idx in vertex_coords]
        elif len(vertex_coords[0]) == 3 or len(vertex_coords[0]) == 2:
            # 3D and 2D
            # print("3D or 2D")
            scallop_poly = []  # [T_inv.mulp(shapes_crs.unproject(vert)) for vert in vertex_coords]
            for vert in vertex_coords:
                pnt_2D = vert.copy()
                pnt_3D = vert.copy()
                pnt_3D.size = 3
                pnt_2D.size = 2
                pnt_3D.z = chunk_elavation.altitude(pnt_2D)
                if pnt_3D is not None and abs(pnt_3D.z) < 100:
                    scallop_poly.append(T_inv.mulp(shapes_crs.unproject(pnt_3D)))
        # elif len(vertex_coords[0]) == 2:
        #     # 2D
        #     # print("2D")
        #     scallop_poly = []
        #     for pnt_2D in vertex_coords:
        #         pnt_3D = pnt_2D.copy()
        #         pnt_3D.size = 3
        #         pnt_3D.z = chunk_elavation.altitude(pnt_2D)
        #         if pnt_3D is not None and abs(pnt_3D.z) < 100:
        #             scallop_poly.append(T_inv.mulp(pnt_3D))
        else:
            print("Unkown shape type")
            continue
        if world_crs:
            scallop_poly = [chunk_transform.mulp(vert) for vert in scallop_poly]
        polygon_up = UpsamplePoly(np.array(scallop_poly), 20)
        try:
            confidence = float(re.findall("\d+\.\d+", shape_label.split('_')[-1])[0])
        except:
            confidence = 1.0
        polygon_entry = [polygon_up, confidence]
        if group_label in chunk_polygons:
            chunk_polygons[group_label].append(polygon_entry)
        else:
            chunk_polygons[group_label] = [polygon_entry]

    # Upsample and check polygon points are valid
    # if len(scallop_poly):
    #     polygon_up = UpsamplePoly(np.array(scallop_poly), 10)
    #     polygon_valid = []
    #     for pnt in polygon_up:
    #         m_pnt = Metashape.Vector(pnt)
    #         ray_1 = m_pnt + Metashape.Vector([0, 0, 1.0]) # T_inv.mulp(m_pnt + Metashape.Vector([0, 0, 1000.0]))
    #         ray_2 = m_pnt - Metashape.Vector([0, 0, 1.0])  # T_inv.mulp(m_pnt - Metashape.Vector([0, 0, 1000.0]))
    #         m_pnt = shapes_crs.unproject(m_pnt)
    #         ray_1 = shapes_crs.unproject(ray_1)
    #         ray_2 = shapes_crs.unproject(ray_2)
    #         pnt_chunk = T_inv.mulp(m_pnt)
    #         ray_1_chunk = T_inv.mulp(ray_1)
    #         ray_2_chunk = T_inv.mulp(ray_2)
    #         closest_intersection = chunk_densepoints.pickPoint(pnt_chunk, ray_1_chunk)
    #         if closest_intersection is None:
    #             closest_intersection = chunk_densepoints.pickPoint(pnt_chunk, ray_2_chunk)
    #         if closest_intersection is not None:
    #             if (m_pnt - closest_intersection).norm() < 0.05 or True:
    #                 polygon_valid.append(np.array(closest_intersection))
    #     if len(polygon_valid) > 2:
    #         chunk_polygons.append(np.array(polygon_valid))

    return chunk_polygons


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