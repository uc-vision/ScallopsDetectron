import Metashape
import numpy as np
import pathlib as p
import os
import pickle
from matplotlib import pyplot as plt
import cv2
from detectron2.structures import BoxMode
from utils import polygon_functions as spf
import json
from tqdm import tqdm
import glob
import math

DISPLAY = True
WAITKEY = 0

SHAPEGROUP_KEY = "ann"

CAM_COV_THRESHOLD = 0.01

CAM_SCALLOP_DIST_THRESH = 2.0
CAM_SCALLOP_Z_THRESH = 2.0

PIX_EDGE_THRESH_CORNER = 10

RECON_BASE = '/csse/research/CVlab/processed_bluerov_data/'  # '/local/ScallopReconstructions/' #'/scratch/data/tkr25/Reconstructions/' #
RECONSTRUCTION_DIRS = [RECON_BASE + '240714-140552/']

def TransformPoints(pnts, transform_quart):
    return np.matmul(transform_quart, np.vstack([pnts, np.ones((1, pnts.shape[1]))]))[:3, :]

def CamVecToPixCoord(pnts_cam, cam_mtx):
    pix_dash = np.matmul(cam_mtx, pnts_cam)
    return (pix_dash / pix_dash[2, :])[:2, :]

if DISPLAY:
    cv2.namedWindow("Annotated img", cv2.WINDOW_GUI_NORMAL)

for RECON_DIR in RECONSTRUCTION_DIRS:
    doc = Metashape.Document()
    doc.open(RECON_DIR + 'recon.psx')
    label_dict = []
    img_dir = 'imgs'
    chunk = doc.chunks[0]

    c = doc.chunks[0].sensors[0].calibration
    cam_fov = np.array([2*math.atan(c.width / (2*(c.f + c.b1))),
                        2*math.atan(c.height / (2*c.f))])

    print("Importing shapes from files")
    shape_files = glob.glob(RECON_DIR + '*anns*.gpkg')
    for shape_file in shape_files:
        chunk.importShapes(shape_file)
        lbl = shape_file.split('/')[-1].split('.')[0]
        chunk.shapes.groups[-1].label = lbl

    chunk_scale = chunk.transform.scale
    print("Chunk scale: {}".format(chunk_scale))
    print("Sorting out polygons...")
    chunk_polygons_d = spf.get_chunk_polygons_dict(chunk, SHAPEGROUP_KEY)
    chunk_polygons = []
    shape_keys = chunk_polygons_d.keys()
    for shape_key in shape_keys:
        if SHAPEGROUP_KEY in shape_key:
            chunk_polygons = chunk_polygons_d[shape_key]
            break
    print("Number of valid scallop polygons in chunk: {}".format(len(chunk_polygons)))

    img_id = 0
    for cam in tqdm(chunk.cameras):
        if cam.transform is None or cam.location_covariance is None or cam.location_covariance.size == 1:
            continue
        cam_quart = np.array(cam.transform).reshape((4, 4))
        cam_cov = np.array(cam.location_covariance).reshape((3, 3))
        xyz_cov_mean = cam_cov[(0, 1, 2), (0, 1, 2)].mean()
        if xyz_cov_mean < CAM_COV_THRESHOLD / chunk_scale:
            img_shape = (cam.image().height, cam.image().width)
            img_path = cam.photo.path
            img_dir, img_fn = img_path.split('/')[-2:]
            c = cam.calibration
            camMtx = np.array([[c.f + c.b1, c.b2, c.cx + c.width / 2],
                               [0, c.f, c.cy + c.height / 2],
                               [0, 0, 1]])
            camDist = np.array([[c.k1, c.k2, c.p2, c.p1, c.k3]])

            #img = np.frombuffer(cam_img_m.tostring(), dtype=np.uint8).reshape((int(cam_img_m.height), int(cam_img_m.width), -1))[:, :, ::-1]
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
            record["file_name"] = img_fn
            record["height"] = height
            record["width"] = width
            record["image_id"] = img_id
            img_id += 1
            objs = []
            display_polygon_l = []
            display_bxs = []

            for scallop_id, (polygon, conf) in enumerate(chunk_polygons):
                polygon_cam = TransformPoints(polygon.T, np.linalg.inv(cam_quart))
                if not spf.in_camera_fov(polygon_cam, cam_fov):
                    continue
                pix_coords = spf.Project2Img(polygon_cam, camMtx, camDist).astype(int).T
                valid_pix_coords = pix_coords[:, np.where((pix_coords[0, :] >= 0) * (pix_coords[0, :] < width) *
                                                          (pix_coords[1, :] >= 0) * (pix_coords[1, :] < height))][:, 0, :]
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
                drawing = np.frombuffer(cam.image().tostring(), dtype=np.uint8).reshape((img_shape[0], img_shape[1], -1))[:, :, ::-1].copy()
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
        with open(RECON_DIR + img_dir + "/" "labels.json", 'w') as fp:
            json.dump(label_dict, fp)