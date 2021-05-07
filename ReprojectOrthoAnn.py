import numpy as np
import pathlib as p
import cv2
import math
from tqdm import tqdm
import json
from detectron2.structures import BoxMode
import Params as P
import pickle


DISPLAY = True

DATASET_DIR = "/local/ScallopMaskDataset/train/"
[path.unlink() for path in p.Path(DATASET_DIR).iterdir()]
with open(P.POLY_ANN_LIST_FN, "rb") as f:
    ann_polys = np.array(pickle.load(f), dtype="object")


def worldPntToCamVec(pnts_wrld, cam_quart):
    ann_vecs_wrld =  pnts_wrld - cam_quart[:3, 3]
    ann_vecs_cam = np.matmul(np.linalg.inv(cam_quart[:3, :3]), ann_vecs_wrld.transpose())
    return ann_vecs_cam

def worldPntToCamAngle(ann_vecs_cam):
    hor_angle = np.sign(ann_vecs_cam[0]) * np.degrees(np.arccos(ann_vecs_cam[2] / np.linalg.norm(ann_vecs_cam[[0, 2], :], axis=0)))
    vert_angle = np.sign(ann_vecs_cam[1]) * np.degrees(np.arccos(ann_vecs_cam[2] / np.linalg.norm(ann_vecs_cam[[1, 2], :], axis=0)))
    return np.vstack([hor_angle, vert_angle])

def CamAngleToPixCoord(angle_xy, im_shape, fov):
    return np.stack([0.5*im_shape[1] * np.tan(np.radians(angle_xy[0])) / np.tan(np.radians(fov[0]/2)) + x0, \
                     0.5*im_shape[0] * np.tan(np.radians(angle_xy[1])) / np.tan(np.radians(fov[1]/2)) + y0]).astype(np.int)


cam_coords = np.load(P.METASHAPE_OUTPUT_DIR + 'cam_coords.npy')
with open(P.METASHAPE_OUTPUT_DIR + 'cam_filenames.txt') as f:
    cam_filenames = f.read().splitlines()
f.close()
camMtx = np.load(P.METASHAPE_OUTPUT_DIR + 'camMtx.npy')
camDist = np.load(P.METASHAPE_OUTPUT_DIR + 'camDist.npy')
x0, y0 = camMtx[:2, 2]
h,  w = (2160, 3840)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camMtx, camDist, (w,h), 0, (w,h))
w_ud, h_ud = roi[2:]
fx = newcameramtx[0, 0]
fy = newcameramtx[1, 1]
FOV = [math.degrees(2*math.atan(w_ud / (2*fx))),
       math.degrees(2*math.atan(h_ud / (2*fy)))]

print("Creating dataset...")
label_dict = []
for idx, (cam_frame, cam_img_path) in tqdm(enumerate(list(zip(cam_coords, cam_filenames))[1000:])):
    img_cam = cv2.imread(cam_img_path)
    img_cam_und = cv2.undistort(img_cam, cameraMatrix=camMtx, distCoeffs=camDist, newCameraMatrix=newcameramtx)
    img_shape = img_cam_und.shape
    img_fn = DATASET_DIR+str(idx)+"_imgUD"+".jpg"
    cv2.imwrite(img_fn, img_cam_und)
    record = {}
    height, width = img_cam_und.shape[:2]
    record["file_name"] = img_fn
    record["height"] = height
    record["width"] = width
    objs = []
    display_polygon_l = []
    display_bxs = []
    for polygon in ann_polys:
        polygon_cam = worldPntToCamVec(polygon, cam_frame)
        valid_cam_vecs = polygon_cam[:, np.where((np.linalg.norm(polygon_cam, axis=0) < 100) * (np.linalg.norm(polygon_cam, axis=0) > 0.0001))][:, 0, :]
        cam_angles = worldPntToCamAngle(valid_cam_vecs)
        valid_angles = cam_angles[:, np.where((np.abs(cam_angles[0, :]) <= FOV[0]/2) * (np.abs(cam_angles[1, :]) <= FOV[1]/2))]
        pix_coords = CamAngleToPixCoord(valid_angles, img_shape, FOV)[:, 0, :]

        if pix_coords.shape[1] < 3:
            continue
        display_polygon_l.append(pix_coords.transpose())
        x_min, y_min = np.min(pix_coords, axis=1).tolist()
        x_max, y_max = np.max(pix_coords, axis=1).tolist()
        display_bxs.append([[x_min, y_min], [x_max, y_max]])
        obj = {
            "bbox": [x_min, y_min, x_max, y_max],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [pix_coords.transpose().tolist()],
            "category_id": 0,
            "iscrowd": 0
        }
        objs.append(obj)
    record["annotations"] = objs
    label_dict.append(record)

    if DISPLAY:
        cv2.imshow("image und proj", cv2.resize(img_cam_und, (1200, 960)))
        drawing = img_cam_und.copy()
        for polygon in display_polygon_l:
            cv2.polylines(drawing, [polygon], False, (0, 255, 0), thickness=2)
        for box_pt1, box_pt2 in display_bxs:
            cv2.rectangle(drawing, tuple(box_pt1), tuple(box_pt2), (0, 255, 255), 2)
        cv2.imshow('Contours', cv2.resize(drawing, (1500, 1500)))
        key = cv2.waitKey()
        if key == ord('b'):
            break
        if key == ord('q'):
            exit(0)

with open(DATASET_DIR + "labels.json", 'w') as fp:
    json.dump(label_dict, fp)
