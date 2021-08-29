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

DISPLAY = True
WRITE = True
UD_ALPHA = 0

DATASET_DIR = "/local/ScallopMaskDataset/train_lr/"
if WRITE:
    [path.unlink() for path in p.Path(DATASET_DIR).iterdir()]
with open(P.POLY_ANN_LIST_FN, "rb") as f:
    ann_polys = pickle.load(f)

def worldPntToCamVec(pnts_wrld, cam_quart):
    ann_vecs_cam = np.matmul(np.linalg.inv(cam_quart), np.vstack([pnts_wrld.transpose(), np.ones((1, pnts_wrld.shape[0]))]))[:3, :]
    return ann_vecs_cam

def CamVecToPixCoord(pnts_cam, cam_mtx):
    pix_dash = np.matmul(cam_mtx, pnts_cam)
    return (pix_dash / pix_dash[2, :])[:2, :]


cam_coords = np.load(P.METASHAPE_OUTPUT_DIR + 'cam_coords.npy')
with open(P.METASHAPE_OUTPUT_DIR + 'cam_filenames.txt') as f:
    cam_filenames = f.read().splitlines()
f.close()
camMtx = np.load(P.METASHAPE_OUTPUT_DIR + 'camMtx.npy')
camDist = np.load(P.METASHAPE_OUTPUT_DIR + 'camDist.npy')
x0, y0 = camMtx[:2, 2]
h,  w = (2160, 3840)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camMtx, camDist, (w,h), UD_ALPHA, (w,h))
print("Cam undistortion ROI: {}".format(roi))
x_ud, y_ud, w_ud, h_ud = roi
fx = newcameramtx[0, 0]
fy = newcameramtx[1, 1]
FOV = [math.degrees(2*math.atan(w_ud / (2*fx))),
       math.degrees(2*math.atan(h_ud / (2*fy)))]
#print(FOV)

print("Creating dataset...")
label_dict = []

region_center = np.load(P.METASHAPE_OUTPUT_DIR + 'ortho_origin.npy')
region_rotation = np.load(P.METASHAPE_OUTPUT_DIR + 'ortho_rotmat.npy')
chunk_translation = np.load(P.METASHAPE_OUTPUT_DIR + 'chunk_trans.npy')
chunk_rotation = np.load(P.METASHAPE_OUTPUT_DIR + 'chunk_rotmat.npy')

fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.set_box_aspect((1, 1, 1))
#ax.set_zlim(-80,0)
#ax = fig.gca(projection='3d')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
cam_positions = cam_coords[:, :3, 3]
ax.scatter3D(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2], color="g")
ax.scatter3D(cam_positions[0, 0], cam_positions[0, 1], cam_positions[0, 2], color="b")
polygons_avgpos_wrld = []
for polygon in ann_polys:
    avg_pos = np.average(polygon, axis=0)
    if abs(avg_pos[2]) < 2000 and abs(avg_pos[2]) > 10:
        polygons_avgpos_wrld.append(avg_pos)
ann_pos_array = np.array(polygons_avgpos_wrld)
#print(ortho_origin)
print(region_center)

print(chunk_translation)
#ann_pos_array = np.matmul(region_rotation.transpose(), ann_pos_array.transpose()).transpose()# + region_center[:, None].transpose()
#ann_pos_array = np.matmul(chunk_rotation, ann_pos_array.transpose()).transpose() + chunk_translation[:, None].transpose()
#ann_pos_array = np.matmul(chunk_rotation, ann_pos_array.transpose()).transpose() + chunk_translation[:, None].transpose()
#ann_pos_array = (np.matmul(ortho_rotation, ann_pos_array.transpose()) + ortho_origin[:, None]).transpose()
#ann_pos_array = ann_pos_array + np.array([-63.99244719, -301.03955114, 0.0])
#ann_pos_array = np.matmul(ortho_rotation.transpose(), ann_pos_array.transpose()).transpose()
ax.scatter3D(ann_pos_array[:, 0], ann_pos_array[:, 1], ann_pos_array[:, 2], color="r")
plt.show()

for idx, (cam_frame, cam_img_path) in tqdm(enumerate(list(zip(cam_coords, cam_filenames)))):
    if not np.array_equal(cam_frame, np.eye(4)):
        img_cam = cv2.imread(cam_img_path)
        img_cam_und = cv2.undistort(img_cam, cameraMatrix=camMtx, distCoeffs=camDist, newCameraMatrix=newcameramtx)
        img_cam_und_roi = img_cam_und[y_ud:y_ud+h_ud, x_ud:x_ud+w_ud]
        img_shape = img_cam_und_roi.shape
        img_fn = DATASET_DIR+str(idx)+"_imgUD"+".jpg"
        if WRITE:
            cv2.imwrite(img_fn, img_cam_und_roi)
        record = {}
        height, width = img_shape[:2]
        record["file_name"] = img_fn
        record["height"] = height
        record["width"] = width
        objs = []
        display_polygon_l = []
        display_bxs = []
        for polygon in ann_polys:
            polygon_cam = worldPntToCamVec(np.array(polygon), cam_frame) #+np.array([5, 0, -0.35])
            valid_cam_vecs = polygon_cam[:, np.where((np.linalg.norm(polygon_cam, axis=0) < 100) * (np.linalg.norm(polygon_cam, axis=0) > 0.0001))][:, 0, :]
            pix_coords = CamVecToPixCoord(valid_cam_vecs, newcameramtx)
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
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        label_dict.append(record)

        if DISPLAY:
            cv2.imshow("image original", cv2.resize(img_cam, (img_cam.shape[1]//2, img_cam.shape[0]//2)))
            drawing = img_cam_und_roi.copy()
            for polygon in display_polygon_l:
                cv2.polylines(drawing, [polygon], False, (0, 255, 0), thickness=2)
            for box_pt1, box_pt2 in display_bxs:
                cv2.rectangle(drawing, tuple(box_pt1), tuple(box_pt2), (0, 255, 255), 2)
            cv2.imshow('Contours', cv2.resize(drawing, (drawing.shape[1]//2, drawing.shape[0]//2)))
            key = cv2.waitKey()
            if key == ord('b'):
                break
            if key == ord('q'):
                exit(0)

if WRITE:
    with open(DATASET_DIR + "labels.json", 'w') as fp:
        json.dump(label_dict, fp)
