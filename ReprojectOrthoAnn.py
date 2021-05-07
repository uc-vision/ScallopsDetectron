import numpy as np
import pathlib as p
import cv2
import math
from tqdm import tqdm
import json
from detectron2.structures import BoxMode
import Params as P


DISPLAY = True

DATASET_DIR = "/local/ScallopMaskDataset/sdgh/"
[path.unlink() for path in p.Path(DATASET_DIR).iterdir()]


ortho_paths = [str(path) for path in p.Path(P.METASHAPE_OUTPUT_DIR + 'ortho/').iterdir()]
orthoann_paths = [str(path) for path in p.Path(P.METASHAPE_OUTPUT_DIR + 'ortho_ann/').iterdir()]
dem_paths = [str(path) for path in p.Path(P.METASHAPE_OUTPUT_DIR + 'dem/').iterdir()]
ortho_paths.sort()
orthoann_paths.sort()
dem_paths.sort()
tile_offsets = np.array([[float(path.split('-')[-2]), -float(path.split('-')[-1][:-4]), 0] for path in ortho_paths])
tile_offsets *= P.PIXEL_SCALE * P.TILE_SIZE
path_tuples = list(zip(ortho_paths, orthoann_paths, dem_paths))
#print(path_tuples)
assert len(ortho_paths) == len(orthoann_paths) and len(ortho_paths) == len(dem_paths)
#TODO: add check that all tile filenames line up

cam_coords = np.load(P.METASHAPE_OUTPUT_DIR + 'cam_coords.npy')
with open(P.METASHAPE_OUTPUT_DIR + 'cam_filenames.txt') as f:
    cam_filenames = f.read().splitlines()
f.close()
ortho_origin = np.load(P.METASHAPE_OUTPUT_DIR + 'ortho_origin.npy')
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


print("Extracting annotation points...")
ann_wrld_pnts = []
for tile_idx, (ortho_path, ann_path, dem_path) in tqdm(enumerate(path_tuples)):
    ortho_tile = cv2.imread(ortho_path)
    ann_tile = cv2.imread(ann_path)
    dem_tile = cv2.imread(dem_path, cv2.IMREAD_ANYDEPTH)
    shape = ann_tile.shape
    idxs = np.where(ann_tile[:, :, 0] != 0)
    try:
        elavations = dem_tile[idxs]
        ann_wrld_pnts.extend(list(np.dstack([idxs[1]*P.PIXEL_SCALE, -idxs[0]*P.PIXEL_SCALE, elavations])[0] +
                                  ortho_origin + tile_offsets[tile_idx]))
    except:
        print("Wrld pnt failed!")
ann_wrld_pnts = np.array(ann_wrld_pnts)

print("Creating dataset...")
CIRCLE_RAD = 10
label_dict = []
for idx, (cam_frame, cam_img_path) in tqdm(enumerate(list(zip(cam_coords, cam_filenames))[1000:])):
    img_cam = cv2.imread(cam_img_path)
    img_cam_und = cv2.undistort(img_cam, cameraMatrix=camMtx, distCoeffs=camDist, newCameraMatrix=newcameramtx)
    ann_img = np.zeros_like(img_cam_und)
    img_shape = img_cam_und.shape
    cam_vecs = worldPntToCamVec(ann_wrld_pnts, cam_frame)
    valid_cam_vecs = cam_vecs[:, np.where((np.linalg.norm(cam_vecs, axis=0) < 100) * (np.linalg.norm(cam_vecs, axis=0) > 0.0001))]
    cam_angles = worldPntToCamAngle(valid_cam_vecs)
    valid_angles = cam_angles[:, np.where((np.abs(cam_angles[0, :]) <= FOV[0]/2) * (np.abs(cam_angles[1, :]) <= FOV[1]/2))]
    pix_coords = CamAngleToPixCoord(valid_angles, img_shape, FOV)
    for pix_coord in pix_coords.transpose()[:, 0]:
        cv2.circle(ann_img, (int(pix_coord[0]), int(pix_coord[1])), CIRCLE_RAD, (255, 255, 255), -1)

    img_fn = DATASET_DIR+str(idx)+"_imgUD"+".jpg"
    cv2.imwrite(img_fn, img_cam_und)

    record = {}
    height, width = img_cam_und.shape[:2]
    record["file_name"] = img_fn
    record["height"] = height
    record["width"] = width
    objs = []
    ann_img_gray = cv2.cvtColor(ann_img, cv2.COLOR_BGR2GRAY)
    ann_img_gray[:, [0, -1]] = 0
    ann_img_gray[[0, -1], :] = 0
    canny_output = cv2.Canny(ann_img_gray, 1, 2)
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_poly = []
    for i, c in enumerate(contours):
        poly = cv2.approxPolyDP(c, 1, True)
        contours_poly.append(poly)
        poly_list = [[int(vert[0]), int(vert[1])] for vert in poly[:, 0]]
        if len(poly_list) < 3:
            continue
        x,y,w,h = cv2.boundingRect(poly)
        obj = {
            "bbox": [x, y, x+w, y+h],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [poly_list],
            "category_id": 0,
            "iscrowd": 0
        }
        objs.append(obj)
    record["annotations"] = objs
    label_dict.append(record)

    if DISPLAY:
        cv2.imshow("image und proj", cv2.resize(0.8*img_cam_und + 0.2*ann_img, (1200, 960))/255)
        cv2.imshow("ann img", cv2.resize(ann_img_gray, (1200, 960)))
        if len(contours_poly):
            drawing = img_cam_und.copy()
            cv2.drawContours(drawing, contours_poly, -1, (0, 255, 0))
            cv2.imshow('Canny', cv2.resize(canny_output, (1500, 1500)))
            cv2.imshow('Contours', cv2.resize(drawing, (1500, 1500)))
        key = cv2.waitKey()
        if key == ord('q'):
            exit(0)

with open(DATASET_DIR + "labels.json", 'w') as fp:
    json.dump(label_dict, fp)
