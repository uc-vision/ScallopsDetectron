import numpy as np
import pathlib as p
import cv2
import math
from tqdm import tqdm
import json
from detectron2.structures import BoxMode
import Params as P
import pickle
import os

DISPLAY = False

try:
    os.mkdir(P.DATASET_DIR)
except OSError as error:
    print(error)

if not DISPLAY:
    [path.unlink() for path in p.Path(P.DATASET_DIR).iterdir()]
with open(P.POLY_ANN_LIST_PATH, "rb") as f:
    ann_polys_wrld = pickle.load(f)

ortho_paths = [str(path) for path in p.Path(P.METASHAPE_OUTPUT_DIR + 'ortho/').iterdir()]
ortho_paths.sort()
tile_offsets = np.array([[float(path.split('-')[-2]), -float(path.split('-')[-1][:-4])] for path in ortho_paths])
tile_offsets *= P.TILE_SIZE
ortho_origin = np.load(P.METASHAPE_OUTPUT_DIR + 'ortho_origin.npy')
pxl_scale = P.PIXEL_SCALE #np.load(P.METASHAPE_OUTPUT_DIR + 'pix_scale.npy')

def UpsamplePoly(polygon, num=10):
    poly_ext = np.append(polygon, [polygon[0, :]], axis=0)
    up_poly = []
    for idx in range(poly_ext.shape[0]-1):
        int_points = np.linspace(poly_ext[idx], poly_ext[idx+1], num=num)
        up_poly.extend(int_points)
    return np.array(up_poly)

def LoadTiles(paths, dims=3, dtype=np.uint8, read_flags=None):
    tile_offsets_rc = np.array([[float(path.split('-')[-1][:-4]), float(path.split('-')[-2])] for path in paths])
    tile_offsets_rc *= P.TILE_SIZE
    tile_extents = tile_offsets_rc + P.TILE_SIZE
    max_extents = np.max(tile_extents, axis=0).astype(int)
    min_extents = np.min(tile_extents, axis=0).astype(int) - P.TILE_SIZE
    ortho_full = np.zeros((max_extents[0]-min_extents[0], max_extents[1]-min_extents[1], dims), dtype=dtype)
    full_shape = ortho_full.shape
    print("Ortho Shape: {}, loading tiles...".format(full_shape))
    for ortho_tile_path, offset in tqdm(list(zip(paths, tile_offsets_rc.astype(int)))):
        ortho_tile = cv2.imread(ortho_tile_path, read_flags)
        tile_shape = ortho_tile.shape
        tile_idx = offset - min_extents
        ortho_full[tile_idx[0]:(tile_idx[0]+tile_shape[0]), tile_idx[1]:(tile_idx[1]+tile_shape[1])] = ortho_tile.reshape((tile_shape[0], tile_shape[1], -1))
    return ortho_full, full_shape, min_extents.astype(float), max_extents.astype(float)
ortho_full, full_shape, min_exts, max_exts = LoadTiles(ortho_paths)
ORTHO_RS = 20
small_ortho = cv2.resize(ortho_full, (full_shape[1]//ORTHO_RS, full_shape[0]//ORTHO_RS))
cv2.imshow("Ortho small", small_ortho)
cv2.waitKey(1)

polys_ortho_pix = []
for polygon in ann_polys_wrld:
    polygon_ortho_m = np.matmul(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), np.array(polygon).transpose() - ortho_origin[:, None]).transpose()
    polygon_ortho_pix = (polygon_ortho_m[:, :2] / pxl_scale).astype(np.int32) - np.array([min_exts[1], min_exts[0]])
    polys_ortho_pix.append(polygon_ortho_pix)

print("Creating dataset...")
label_dict = []
num_subs_x = math.ceil(full_shape[1] / (P.ORTHOSUB_SHAPE[1] - P.ORTHOSUB_OVERLAP_PIX))
num_subs_y = math.ceil(full_shape[0] / (P.ORTHOSUB_SHAPE[0] - P.ORTHOSUB_OVERLAP_PIX))
if DISPLAY:
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Annotated", cv2.WINDOW_NORMAL)
img_id = 0
data_pix_u_sum = np.array([0.0, 0.0, 0.0])
for row_idx in tqdm(range(num_subs_y)):
    for col_idx in range(num_subs_x):
        sub_idx_x = int(col_idx*(P.ORTHOSUB_SHAPE[1] - P.ORTHOSUB_OVERLAP_PIX))
        sub_idx_y = int(row_idx*(P.ORTHOSUB_SHAPE[0] - P.ORTHOSUB_OVERLAP_PIX))
        ortho_sub_img = ortho_full[sub_idx_y:min(sub_idx_y+P.ORTHOSUB_SHAPE[0], full_shape[0]),
                        sub_idx_x:min(sub_idx_x+P.ORTHOSUB_SHAPE[1], full_shape[1]), :]
        if (ortho_sub_img == 0).all():
            continue
        img_fn = "ortho_" + str(row_idx) + "_" + str(col_idx) + ".png" #P.DATASET_DIR +
        if not DISPLAY:
            cv2.imwrite(P.DATASET_DIR + img_fn, ortho_sub_img)
        data_pix_u_sum += np.mean(ortho_sub_img, axis=(0, 1))
        record = {}
        height, width, _ = ortho_sub_img.shape
        record["file_name"] = img_fn
        record["height"] = height
        record["width"] = width
        record["image_id"] = img_id
        img_id += 1
        objs = []
        display_polygon_l = []
        display_bxs = []
        for scallop_id, poly in enumerate(polys_ortho_pix):
            poly = UpsamplePoly(poly)
            poly_subimg = (poly - np.array([sub_idx_x, sub_idx_y])).transpose()
            valid_pix_coords = poly_subimg[:, np.where((poly_subimg[0, :] >= 0) * (poly_subimg[0, :] < width) *
                                                      (poly_subimg[1, :] >= 0) * (poly_subimg[1, :] < height))][:, 0, :].astype(np.int)
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

        if DISPLAY and len(display_polygon_l):
            drawing = ortho_sub_img.copy()
            for polygon in display_polygon_l:
                cv2.polylines(drawing, [polygon], False, (0, 255, 0), thickness=2)
            for box_pt1, box_pt2 in display_bxs:
                cv2.rectangle(drawing, tuple(box_pt1), tuple(box_pt2), (0, 255, 255), 2)
            cv2.imshow("Original", ortho_sub_img)
            cv2.imshow('Annotated', drawing)
            key = cv2.waitKey()
            if key == ord('b'):
                break
            if key == ord('q'):
                exit(0)

print("Data pixel means BGR: {}".format(data_pix_u_sum / (img_id + 1)))

if not DISPLAY:
    with open(P.DATASET_DIR + "labels.json", 'w') as fp:
        json.dump(label_dict, fp)
