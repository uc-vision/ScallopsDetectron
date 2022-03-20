import Metashape
import numpy as np
import pathlib as p
import os
import pickle
from matplotlib import pyplot as plt
import Params as P
import cv2
from detectron2.structures import BoxMode
import json
from tqdm import tqdm

DISPLAY = False
WAITKEY = 0
UD_ALPHA = 0

# RECONSTRUCTION_DIRS = [P.METASHAPE_OUTPUT_BASE + 'gopro_116_1/',
#                        P.METASHAPE_OUTPUT_BASE + 'gopro_118/',
#                        P.METASHAPE_OUTPUT_BASE + 'gopro_123/',
#                        P.METASHAPE_OUTPUT_BASE + 'gopro_124/',
#                        P.METASHAPE_OUTPUT_BASE + 'gopro_125/']

DATASET_DIR_BASE = "/local/ScallopMaskDataset/"
METASHAPE_OUTPUT_BASE = '/local/ScallopReconstructions/'
RECONSTRUCTION_DIRS = [METASHAPE_OUTPUT_BASE + 'gopro_119/']

def TransformPoints(pnts, transform_quart):
    return np.matmul(transform_quart, np.vstack([pnts, np.ones((1, pnts.shape[1]))]))[:3, :]

def CamVecToPixCoord(pnts_cam, cam_mtx):
    pix_dash = np.matmul(cam_mtx, pnts_cam)
    return (pix_dash / pix_dash[2, :])[:2, :]

def UpsamplePoly(polygon, num=10):
    poly_ext = np.append(polygon, [polygon[0, :]], axis=0)
    up_poly = []
    for idx in range(poly_ext.shape[0]-1):
        int_points = np.linspace(poly_ext[idx], poly_ext[idx+1], num=num)
        up_poly.extend(int_points)
    return np.array(up_poly)

if DISPLAY:
    cv2.namedWindow("Annotated img", cv2.WINDOW_GUI_NORMAL)

for RECON_DIR in RECONSTRUCTION_DIRS:
    DATASET_DIRECTORY = DATASET_DIR_BASE + RECON_DIR[:-1].split('/')[-1] + '_prop/'
    try:
        os.mkdir(DATASET_DIRECTORY)
    except OSError as error:
        print(error)
    if not DISPLAY:
        [path.unlink() for path in p.Path(DATASET_DIRECTORY).iterdir()]

    doc = Metashape.Document()
    doc.open(RECON_DIR + 'recon.psx')
    label_dict = []

    for chunk_idx, chunk in enumerate(doc.chunks):
        print("Reprojecting chunk {}...".format(chunk_idx))
        chunk_marker_dict = {v.key: v.position for v in chunk.markers}
        chunk_densepoints = chunk.dense_cloud
        chunk_transform = chunk.transform.matrix
        chunk_scale = chunk.transform.scale
        print("Sorting out polygons...")
        chunk_polygons = []
        for shape in chunk.shapes.shapes:
            if shape.type == Metashape.Shape.Type.Polygon:
                polygon = []
                vertex_coords = shape.geometry.coordinates[0]
                if shape.has_z:
                    if isinstance(vertex_coords[0], int):
                        polygon = [chunk_marker_dict[idx] for idx in vertex_coords]
                    else:
                        polygon = [pnt for pnt in vertex_coords]
                else:
                    polygon = []
                    for pnt_2D in vertex_coords:
                        vec = pnt_2D
                        vec.size = 3
                        vec.z = 0
                        vec_pnt = vec+Metashape.Vector([0, 0, 1])
                        vec = chunk_transform.inv().mulp(vec)
                        vec_pnt = chunk_transform.inv().mulp(vec_pnt)
                        pnt_3D = chunk_densepoints.pickPoint(vec, vec_pnt)
                        if pnt_3D is not None and abs(pnt_3D.z) < 100:
                            polygon.append(pnt_3D)

                # Upsample and check polygon points are valid
                if len(polygon):
                    polygon_up = UpsamplePoly(np.array(polygon), 5)
                    polygon_valid = []
                    for pnt in polygon_up:
                        m_pnt = Metashape.Vector(pnt)
                        ray_pnt = Metashape.Vector([0, 0, -0.1])
                        closest_intersection = chunk_densepoints.pickPoint(m_pnt, ray_pnt)
                        if closest_intersection is not None:
                            if (m_pnt - closest_intersection).norm() < 0.1:
                                polygon_valid.append(pnt)
                    if len(polygon_valid) > 2:
                        chunk_polygons.append(np.array(polygon_valid))
        print("Number of valid scallop polygons in chunk: {}".format(len(chunk_polygons)))

        img_id = 0
        for cam in tqdm(chunk.cameras):
            cam_quart = np.array(cam.transform).reshape((4, 4))
            if not np.array_equal(cam_quart, np.eye(4)):
                cam_img_m = cam.image()
                img_path = cam.photo.path
                img_fn = img_path.split('/')[-1]
                c = cam.calibration
                camMtx = np.array([[c.f + c.b1, c.b2, c.cx + c.width / 2],
                              [0, c.f, c.cy + c.height / 2],
                              [0, 0, 1]])
                camDist = np.array([[c.k1, c.k2, c.p2, c.p1, c.k3]])
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camMtx, camDist, (int(cam_img_m.width), int(cam_img_m.height)),
                                                                  UD_ALPHA, (int(cam_img_m.width), int(cam_img_m.height)))
                x_ud, y_ud, w_ud, h_ud = roi
                img = np.frombuffer(cam_img_m.tostring(), dtype=np.uint8).reshape((int(cam_img_m.height), int(cam_img_m.width), -1))[:, :, ::-1]
                img_cam_und = cv2.undistort(img, cameraMatrix=camMtx, distCoeffs=camDist, newCameraMatrix=newcameramtx)
                img_cam_und_roi = img_cam_und[y_ud:y_ud+h_ud, x_ud:x_ud+w_ud].copy()

                # Deletion of image regions not in orthomosaic
                valid_pixel_mask = np.zeros_like(img_cam_und_roi)
                row_indices, col_indices = np.indices((h_ud, w_ud))[:, ::50, ::50]
                pnts_uv = np.array([col_indices.flatten(), row_indices.flatten(), col_indices.size*[1]], dtype=np.float32)
                cam_rays = np.matmul(np.linalg.inv(camMtx), pnts_uv)
                cam_pnts = TransformPoints(cam_rays, cam_quart)
                for cam_pnt in cam_pnts.transpose():
                    cam_origin = Metashape.Vector(cam_quart[:3, 3])
                    ray_pnt = Metashape.Vector(cam_pnt)
                    closest_intersection = chunk_densepoints.pickPoint(cam_origin, ray_pnt)
                    if closest_intersection is not None:
                        if (cam_origin - closest_intersection).norm() < 10:
                            img_pixels = cam.project(closest_intersection)
                            if img_pixels is not None:
                                valid_pix_int = np.array(img_pixels, dtype=np.int).reshape((2,))
                                cv2.circle(valid_pixel_mask, tuple(valid_pix_int), 100, (1, 1, 1), -1)
                img_cam_und_roi *= valid_pixel_mask

                if not DISPLAY:
                    cv2.imwrite(DATASET_DIRECTORY + img_fn, img_cam_und_roi)

                record = {}
                height, width = img_cam_und_roi.shape[:2]
                record["file_name"] = img_fn
                record["height"] = height
                record["width"] = width
                record["image_id"] = img_id
                img_id += 1
                objs = []
                display_polygon_l = []
                display_bxs = []

                for scallop_id, polygon in enumerate(chunk_polygons):
                    polygon_cam = TransformPoints(polygon.transpose(), np.linalg.inv(cam_quart))
                    pix_coords = CamVecToPixCoord(polygon_cam, newcameramtx)
                    pix_coords -= [[x_ud], [y_ud]]
                    valid_pix_coords = pix_coords[:, np.where((pix_coords[0, :] >= 0) * (pix_coords[0, :] < w_ud) *
                                                              (pix_coords[1, :] >= 0) * (pix_coords[1, :] < h_ud))][:, 0, :].astype(np.int)
                    if valid_pix_coords.shape[1] < 3:
                        continue

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

                if DISPLAY:
                    drawing = img_cam_und_roi.copy()
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
        with open(DATASET_DIRECTORY + "labels.json", 'w') as fp:
            json.dump(label_dict, fp)