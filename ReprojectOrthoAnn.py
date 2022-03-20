import numpy as np
import pathlib as p
import cv2
import math
from tqdm import tqdm
import json
from detectron2.structures import BoxMode
import Params as P
import pickle
from matplotlib import pyplot as plt
import os

MASK_ORTHO_VALID = True
DISPLAY = False
GRAY = True
DISP_DS = 1
UD_ALPHA = 1

DATASET_DIRECTORY = P.DATASET_DIR[:-1] + "_prop" + '_bw/' if GRAY else '/'

try:
    os.mkdir(DATASET_DIRECTORY)
except OSError as error:
    print(error)

if not DISPLAY:
    [path.unlink() for path in p.Path(DATASET_DIRECTORY).iterdir()]
with open(P.POLY_ANN_LIST_PATH, "rb") as f:
    ann_polys = pickle.load(f)
with open(P.VALID_ORTHO_POINTS_PATH, "rb") as f:
    valid_ortho_centers = pickle.load(f)

def worldPntToCamVec(pnts_wrld, cam_quart):
    ann_vecs_cam = np.matmul(np.linalg.inv(cam_quart), np.vstack([pnts_wrld.transpose(), np.ones((1, pnts_wrld.shape[0]))]))[:3, :]
    return ann_vecs_cam

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

cam_coords = np.load(P.METASHAPE_OUTPUT_DIR + 'cam_coords.npy')
with open(P.METASHAPE_OUTPUT_DIR + 'cam_filenames.txt') as f:
    cam_filenames = f.read().splitlines()
f.close()
camMtx = np.load(P.METASHAPE_OUTPUT_DIR + 'camMtx.npy')
camDist = np.load(P.METASHAPE_OUTPUT_DIR + 'camDist.npy')
x0, y0 = camMtx[:2, 2]
img_sample = cv2.imread(cam_filenames[0])
print(cam_filenames[0])
h, w, c = img_sample.shape
print(img_sample.shape)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camMtx, camDist, (w,h), UD_ALPHA, (w,h))
print("Cam undistortion ROI: {}".format(roi))
x_ud, y_ud, w_ud, h_ud = roi
fx = newcameramtx[0, 0]
fy = newcameramtx[1, 1]
FOV = [math.degrees(2*math.atan(w_ud / (2*fx))),
       math.degrees(2*math.atan(h_ud / (2*fy)))]
print("Camera FOV: {}".format(FOV))

print("Creating dataset...")
label_dict = []

fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.set_zlim(-80,0)
#ax = fig.gca(projection='3d')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
cam_positions = cam_coords[:, :3, 3]
ax.scatter3D(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2], color="g")
ax.scatter3D(cam_positions[0, 0], cam_positions[0, 1], cam_positions[0, 2], color="b")

polygons_avgpos_wrld = []
new_ann_polys = []
for polygon in ann_polys:
    poly = polygon# + np.array([0, 0, 1.145])
    avg_pos = np.average(poly, axis=0)
    new_ann_polys.append(poly)
    polygons_avgpos_wrld.append(avg_pos)
ann_pos_array = np.array(polygons_avgpos_wrld)
ann_pos_array = ann_pos_array[np.where(np.abs(ann_pos_array[:, 2]) < 100)]
ann_polys = new_ann_polys

if DISPLAY:
    ax.scatter3D(ann_pos_array[:, 0], ann_pos_array[:, 1], ann_pos_array[:, 2], color="r")
    DS = 1000
    ax.scatter3D(valid_ortho_centers[0][::DS], valid_ortho_centers[1][::DS], valid_ortho_centers[2][::DS], color="m")
    plt.show()

    cv2.namedWindow("image original", cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow("image UD", cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow("Contours", cv2.WINDOW_GUI_NORMAL)

img_id = 0
for idx, (cam_frame, cam_img_path) in tqdm(enumerate(list(zip(cam_coords, cam_filenames)))):
    if not np.array_equal(cam_frame, np.eye(4)):
        img_cam = cv2.imread(cam_img_path)
        unique_vals, counts = np.unique(img_cam[::2, ::2, :], return_counts=True)
        if unique_vals[np.argmax(counts)] == 0:
            print("Partial frame!")
            continue
        if cam_frame[2, 2] > 0:
            print("Invalid cam z-axis!")
            continue
        img_cam_und = cv2.undistort(img_cam, cameraMatrix=camMtx, distCoeffs=camDist, newCameraMatrix=newcameramtx)
        img_cam_und_roi = img_cam_und[y_ud:y_ud+h_ud, x_ud:x_ud+w_ud].copy()
        img_shape = img_cam_und_roi.shape
        img_fn = str(idx)+"_imgUD"+".jpg"

        if MASK_ORTHO_VALID:
            valid_pixels_cam = worldPntToCamVec(valid_ortho_centers.transpose(), cam_frame)
            valid_pixels_cam = valid_pixels_cam[:, np.where((np.linalg.norm(valid_pixels_cam, axis=0) < 10) * (np.linalg.norm(valid_pixels_cam, axis=0) > 0.01))][:, 0, :]
            pix_coords = CamVecToPixCoord(valid_pixels_cam, newcameramtx)
            pix_coords -= [[x_ud], [y_ud]]
            valid_pix_coords = pix_coords[:, np.where((pix_coords[0, :] >= 0) * (pix_coords[0, :] < w_ud) *
                                                      (pix_coords[1, :] >= 0) * (pix_coords[1, :] < h_ud))][:, 0, :].astype(np.int)
            valid_in_ortho_mask = np.zeros_like(img_cam_und_roi)
            for pix in valid_pix_coords.transpose():
                cv2.circle(valid_in_ortho_mask, tuple(pix), 20, (1, 1, 1), -1)
            img_cam_und_roi *= valid_in_ortho_mask

        if not DISPLAY:
            if GRAY:
                img_cam_und_roi = cv2.cvtColor(img_cam_und_roi, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(DATASET_DIRECTORY + img_fn, img_cam_und_roi)
        record = {}
        height, width = img_shape[:2]
        record["file_name"] = img_fn
        record["height"] = height
        record["width"] = width
        record["image_id"] = img_id
        img_id += 1
        objs = []
        display_polygon_l = []
        display_bxs = []

        for scallop_id, polygon in enumerate(ann_polys):
            polygon = UpsamplePoly(polygon)
            polygon_cam = worldPntToCamVec(np.array(polygon), cam_frame)
            valid_cam_vecs = polygon_cam[:, np.where((np.linalg.norm(polygon_cam, axis=0) < 3) * (np.linalg.norm(polygon_cam, axis=0) > 0.01))][:, 0, :]
            pix_coords = CamVecToPixCoord(valid_cam_vecs, newcameramtx)
            pix_coords -= [[x_ud], [y_ud]]
            valid_pix_coords = pix_coords[:, np.where((pix_coords[0, :] >= 0) * (pix_coords[0, :] < w_ud) *
                                                      (pix_coords[1, :] >= 0) * (pix_coords[1, :] < h_ud))][:, 0, :].astype(np.int)
            if valid_pix_coords.shape[1] < 3:
                continue
            # if not valid_pix_coords.shape[1] % 2 == 0:
            #     valid_pix_coords = np.append(valid_pix_coords, valid_pix_coords[:, 0][:, None], axis=1)
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
            drawing = img_cam_und_roi.copy()
            for polygon in display_polygon_l:
                cv2.polylines(drawing, [polygon], False, (0, 255, 0), thickness=2)
            for box_pt1, box_pt2 in display_bxs:
                cv2.rectangle(drawing, tuple(box_pt1), tuple(box_pt2), (0, 255, 255), 2)
            cv2.imshow("image original", cv2.resize(img_cam, (img_cam.shape[1]//DISP_DS, img_cam.shape[0]//DISP_DS)))
            cv2.imshow("image UD", cv2.resize(img_cam_und, (img_cam_und.shape[1]//DISP_DS, img_cam_und.shape[0]//DISP_DS)))
            cv2.imshow('Contours', cv2.resize(drawing, (drawing.shape[1]//DISP_DS, drawing.shape[0]//DISP_DS)))
            key = cv2.waitKey()
            if key == ord('b'):
                break
            if key == ord('q'):
                exit(0)

if not DISPLAY:
    with open(DATASET_DIRECTORY + "labels.json", 'w') as fp:
        json.dump(label_dict, fp)
