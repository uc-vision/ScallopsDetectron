import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
from detectron2.structures import BoxMode
from utils import polygon_functions as spf
from utils import vpz_utils, tiff_utils, geo_utils
import geopandas as gpd
import pickle
import json
from tqdm import tqdm
import glob
import math
from shapely.geometry import *

DISPLAY = False
WAITKEY = 0

CAM_COV_THRESHOLD = 0.02

CAM_SCALLOP_DIST_THRESH = 2.0
CAM_SCALLOP_Z_THRESH = 2.0

PIX_EDGE_THRESH_CORNER = 10

PROCESSED_BASEDIR = '/csse/research/CVlab/processed_bluerov_data/'
DONE_DIRS_FILE = PROCESSED_BASEDIR + 'dirs_done.txt'

def TransformPoints(pnts, transform_quart):
    return np.matmul(transform_quart, np.vstack([pnts, np.ones((1, pnts.shape[1]))]))[:3, :]

def CamVecToPixCoord(pnts_cam, cam_mtx):
    pix_dash = np.matmul(cam_mtx, pnts_cam)
    return (pix_dash / pix_dash[2, :])[:2, :]


def create_dl_file(data_dir):
    if DISPLAY:
        cv2.namedWindow("Annotated img", cv2.WINDOW_GUI_NORMAL)

    print("Loading Chunk Telemetry")
    with open(data_dir + "chunk_telemetry.pkl", "rb") as pkl_file:
        chunk_telem = pickle.load(pkl_file)
        chunk_scale = chunk_telem['0']['scale']
        chunk_transform = chunk_telem['0']['transform']
        chunk_inv_transform = np.linalg.inv(chunk_transform)

    print("Loading Camera Telemetry")
    with open(data_dir + "camera_telemetry.pkl", "rb") as pkl_file:
        camera_telem = pickle.load(pkl_file)

    print("Importing shapes from gpkgs and .vpz")
    # TODO: check for annotations overlap??
    shape_layers = []
    shape_fpaths = glob.glob(data_dir + '*Poly*.gpkg')
    for shape_path in shape_fpaths:
        shape_gdf = gpd.read_file(shape_path)
        shapes_label = shape_path.split('/')[-1].split('.')[0]
        shape_layers.append(shape_gdf)
    # shape_layers_vpz = vpz_utils.get_shape_layers_gpd(data_dir, data_dir.split('/')[-2] + '.vpz')
    # for shape_layer in shape_layers_vpz:
    #     if 'Poly' in shape_layer[0]:
    #         shape_layers.append(shape_layer[1])

    print("Loading DEMs")
    dem_obj = tiff_utils.DEM(data_dir + 'geo_tiffs/')
    # Transformer from geographic/geodetic coordinates to geocentric
    ccs2gcs = lambda pnt: geo_utils.geodetic_to_geocentric(pnt[1], pnt[0], pnt[2])
    print("Getting 3D polygons using DEMs")
    polygons_chunk = []
    for shape_gdf in shape_layers:
        for i, row in shape_gdf.iterrows():
            if isinstance(row.geometry, Polygon):
                poly_2d = np.array(row.geometry.exterior.coords)[:, :2]
                poly_3d = dem_obj.get_3d_polygon(poly_2d)
                poly_3d_geocentric = np.apply_along_axis(ccs2gcs, 1, poly_3d)
                poly_3d_chunk = TransformPoints(poly_3d_geocentric.T, chunk_inv_transform).T

                eig_vecs, eig_vals, center_pnt = spf.pca(poly_3d_chunk)
                poly_principal = np.matmul(np.linalg.inv(eig_vecs), (poly_3d_chunk - center_pnt).T)

                # z_dist = np.abs(poly_principal[2]) * chunk_scale
                # if np.max(z_dist) > 0.01:
                #     print(z_dist)

                # TODO: flatten polygon elevation values - PCA doesnt work when there is alot of noise

                poly_principal[2] = 0.0
                poly_3d_chunk = np.matmul(eig_vecs, poly_principal).T + center_pnt

                poly_3d_chunk_up = spf.UpsamplePoly(poly_3d_chunk, 20)
                polygons_chunk.append(poly_3d_chunk_up)

    print("Number of valid scallop polygons in chunk: {}".format(len(polygons_chunk)))

    img_id = 0
    label_dict = []
    for cam_label, cam_telem in tqdm(camera_telem.items()):
        cam_quart = cam_telem['q44']
        cam_cov = cam_telem['loc_cov33']
        xyz_cov_mean = cam_cov[(0, 1, 2), (0, 1, 2)].mean()
        if xyz_cov_mean < CAM_COV_THRESHOLD / chunk_scale:
            img_shape = cam_telem['shape']
            img_path = cam_telem['fpath']
            img_path_rel = '/'.join(img_path.split('/')[-3:])
            camMtx = cam_telem['cam_mtx']
            camDist = cam_telem['cam_dist']
            cam_fov = cam_telem['cam_fov']

            # img = np.frombuffer(cam_img_m.tostring(), dtype=np.uint8).reshape((int(cam_img_m.height), int(cam_img_m.width), -1))[:, :, ::-1]
            # img_cam_und = cv2.undistort(img, cameraMatrix=camMtx, distCoeffs=camDist, newCameraMatrix=newcameramtx)

            # Deletion of image regions not in orthomosaic
            # valid_pixel_mask = np.zeros_like(img)
            # row_indices, col_indices = np.indices((img_shape[0], img_shape[1]))[:, ::20, ::20]
            # pnts_uv = np.array([col_indices.flatten(), row_indices.flatten(), col_indices.size*[1]], dtype=np.float32)
            # cam_rays = np.matmul(np.linalg.inv(camMtx), pnts_uv)
            # cam_pnts = TransformPoints(cam_rays, cam_quart)
            # for cam_pnt in cam_pnts.transpose():
            #     cam_origin = Metashape.Vector(cam_quart[:3, 3])
            #     ray_pnt = Metashape.Vector(cam_pnt)
            #     closest_intersection = chunk_densepoints.pickPoint(cam_origin, ray_pnt)
            #     if closest_intersection is not None:
            #         if (cam_origin - closest_intersection).norm() < 10:
            #             img_pixels = cam.project(closest_intersection)
            #             if img_pixels is not None:
            #                 valid_pix_int = np.array(img_pixels, dtype=np.int).reshape((2,))
            #                 cv2.circle(valid_pixel_mask, tuple(valid_pix_int), 100, (1, 1, 1), -1)
            # img_cam_und_roi *= valid_pixel_mask

            record = {}
            height, width = img_shape
            record["file_name"] = img_path_rel
            record["height"] = height
            record["width"] = width
            record["image_id"] = img_id
            img_id += 1
            objs = []
            display_polygon_l = []
            display_bxs = []

            for scallop_id, polygon in enumerate(polygons_chunk):
                polygon_cam = TransformPoints(polygon.T, np.linalg.inv(cam_quart))
                if not spf.in_camera_fov(polygon_cam, cam_fov):
                    continue
                pix_coords = spf.Project2Img(polygon_cam, camMtx, camDist).astype(int).T
                valid_pix_coords = pix_coords[:, np.where((pix_coords[0, :] >= 0) * (pix_coords[0, :] < width) *
                                                          (pix_coords[1, :] >= 0) * (pix_coords[1, :] < height))][:, 0,
                                   :]
                if valid_pix_coords.shape[1] < 3:
                    continue

                # Fill in corner detections where it would cut
                is_edge = np.less(valid_pix_coords, PIX_EDGE_THRESH_CORNER) + \
                          np.greater(valid_pix_coords, np.array([[width - PIX_EDGE_THRESH_CORNER],
                                                                 [height - PIX_EDGE_THRESH_CORNER]]))
                needs_corner = np.logical_xor(is_edge[0, :-1], is_edge[0, 1:]) * \
                               np.logical_xor(is_edge[1, :-1], is_edge[1, 1:]) * \
                               np.logical_xor(is_edge[0, :-1], is_edge[1, :-1])
                if np.any(needs_corner):
                    corner_index = np.argmax(needs_corner) + 1
                    corner_pix = valid_pix_coords[:, corner_index]
                    corner_pix = np.round(corner_pix / np.array([width, height])) * np.array([width, height])
                    valid_pix_coords = np.insert(valid_pix_coords, corner_index, corner_pix, axis=1)

                display_polygon_l.append(valid_pix_coords.transpose())
                x_min, y_min = np.min(valid_pix_coords, axis=1).tolist()
                x_max, y_max = np.max(valid_pix_coords, axis=1).tolist()
                display_bxs.append([[x_min, y_min], [x_max, y_max]])
                obj = {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [valid_pix_coords.transpose().tolist()],
                    "category_id": 0,
                    "id": scallop_id,
                    "iscrowd": 0,
                    "name": 'scallop',
                }
                objs.append(obj)
            record["annotations"] = objs
            label_dict.append(record)

            if DISPLAY and len(display_polygon_l):
                drawing = cv2.imread(img_path)
                for polygon in display_polygon_l:
                    cv2.polylines(drawing, [polygon], False, (0, 255, 0), thickness=2)
                for box_pt1, box_pt2 in display_bxs:
                    cv2.rectangle(drawing, tuple(box_pt1), tuple(box_pt2), (0, 255, 255), 2)
                cv2.imshow("Annotated img", drawing)
                key = cv2.waitKey(WAITKEY)
                if key == ord('b'):
                    break
                if key == ord('q'):
                    exit(0)

    if not DISPLAY:
        print("Saving annotations json...")
        with open(data_dir + "scallop_dataset.json", 'w') as fp:
            json.dump(label_dict, fp)


if __name__ == '__main__':
    with open(DONE_DIRS_FILE, 'r') as todo_file:
        data_dirs = todo_file.readlines()
    for data_dir in data_dirs:
        if 'STOP' in data_dir:
            break
        # Check if this is a valid directory that needs processing
        if len(data_dir) == 1 or '#' in data_dir:
            continue
        data_dir = data_dir.split(' ')[0]
        assert len(data_dir) == 14

        # Process this directory
        create_dl_file(PROCESSED_BASEDIR + data_dir[:-1]+'/')

        break

