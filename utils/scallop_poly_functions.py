import numpy as np
import cv2
from tqdm import tqdm
import Metashape

def get_chunk_polygons_dict(chunk, key=None, world_crs=False):
    chunk_marker_dict = {v.key: v.position for v in chunk.markers}
    chunk_elavation = chunk.elevation
    chunk_transform = chunk.transform.matrix
    T_inv = chunk_transform.inv()
    shapes_crs = chunk.shapes.crs
    assert chunk.marker_crs is None

    chunk_polygons = {}
    for shape in tqdm(chunk.shapes.shapes):
        label = shape.group.label
        if key is not None and key not in label:
            continue
        if shape.geometry.type != Metashape.Geometry.Type.PolygonType:
            continue
        vertex_coords = shape.geometry.coordinates[0]
        if isinstance(vertex_coords[0], int):
            # Markers
            scallop_poly = [chunk_marker_dict[idx] for idx in vertex_coords]
        elif len(vertex_coords[0]) == 3:
            # 3D
            scallop_poly = [T_inv.mulp(shapes_crs.unproject(vert)) for vert in vertex_coords]
        elif len(vertex_coords[0]) == 2:
            # 2D
            scallop_poly = []
            for pnt_2D in vertex_coords:
                pnt_3D = pnt_2D.copy()
                pnt_3D.size = 3
                pnt_3D.z = chunk_elavation.altitude(pnt_2D)
                if pnt_3D is not None and abs(pnt_3D.z) < 100:
                    scallop_poly.append(T_inv.mulp(pnt_3D))
        else:
            print("Unkown shape type")
            continue
        if world_crs:
            scallop_poly = [chunk_transform.mulp(vert) for vert in scallop_poly]
        polygon_up = UpsamplePoly(np.array(scallop_poly), 20)
        if label in chunk_polygons:
            chunk_polygons[label].append(polygon_up)
        else:
            chunk_polygons[label] = [polygon_up]

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