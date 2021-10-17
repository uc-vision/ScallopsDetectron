import numpy as np
import pathlib as p
import cv2
from tqdm import tqdm
import Params as P
import pickle
import math

ortho_paths = [str(path) for path in p.Path(P.METASHAPE_OUTPUT_DIR + 'ortho/').iterdir()]
dem_paths = [str(path) for path in p.Path(P.METASHAPE_OUTPUT_DIR + 'dem/').iterdir()]
dem_paths.sort()
ortho_paths.sort()
assert len(ortho_paths) == len(dem_paths)

path_tuples = list(zip(ortho_paths, dem_paths))
tile_offsets = np.array([[float(path.split('-')[-2]), -float(path.split('-')[-1][:-4])] for path in ortho_paths])
tile_offsets *= P.TILE_SIZE
ortho_origin = np.load(P.METASHAPE_OUTPUT_DIR + 'ortho_origin.npy')

cam_coords = np.load(P.METASHAPE_OUTPUT_DIR + 'cam_coords.npy')
with open(P.METASHAPE_OUTPUT_DIR + 'cam_filenames.txt') as f:
    cam_filenames = f.read().splitlines()

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

ORTHO_RS = 10
ortho_full, full_shape, min_exts, max_exts = LoadTiles(ortho_paths)
small_ortho = cv2.resize(ortho_full, (full_shape[1]//ORTHO_RS, full_shape[0]//ORTHO_RS))
dem_full, shape, min_exts, max_exts = LoadTiles(dem_paths, dims=1, read_flags=cv2.IMREAD_ANYDEPTH, dtype=np.float32)
small_dem_gray = cv2.resize(dem_full, (full_shape[1]//ORTHO_RS, full_shape[0]//ORTHO_RS))[:, :, None]
DEM_MAX = 0.5
DEM_MIN = 0
small_dem = cv2.cvtColor((255 * (small_dem_gray - DEM_MIN) / (DEM_MAX - DEM_MIN)).astype(np.uint8), cv2.COLOR_GRAY2BGR)

key = ''
poly_vert_list = []
def mouse_event(event, x, y, flags, param):
    global poly_vert_list, ortho_polygons, sub_idx_x, sub_idx_y
    if event==cv2.EVENT_RBUTTONDOWN:
        poly_vert_list.append([x, y])
    if event==cv2.EVENT_LBUTTONDOWN:
        closest_poly_idx = -1
        closest_center_dist = 10000
        for idx, poly in enumerate(ortho_polygons):
            poly_cent_wrld = np.average(np.array(poly), axis=0)
            poly_cent_pixels = poly_cent_wrld[:2] / P.PIXEL_SCALE
            poly_cent_pixels -= np.array([sub_idx_x+int(min_exts[1]), sub_idx_y+int(min_exts[0])])
            cent_dist = np.linalg.norm(np.array([x, y]) - poly_cent_pixels)
            if cent_dist < closest_center_dist:
                closest_center_dist = cent_dist
                closest_poly_idx = idx
        if closest_poly_idx != -1 and closest_center_dist < 200:
            ortho_polygons.pop(closest_poly_idx)
            print("Deleted Annotation IDX: {}".format(closest_poly_idx))

ortho_polygons_wrld = []
try:
    with open(P.POLY_ANN_LIST_PATH, "rb") as f:
        ortho_polygons_wrld = pickle.load(f)
except:
    print("Can't open existing annotations file!")

ortho_polygons = []
for polygon in ortho_polygons_wrld:
    polygon_offset = np.matmul(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), np.array(polygon).transpose() - ortho_origin[:, None]).transpose()
    #polygon_offset /= 0.002
    #polygon_offset *= 0.0005
    ortho_polygons.append(polygon_offset)

cv2.namedWindow("ortho_ann", cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow("Small Ortho", cv2.WINDOW_GUI_NORMAL)
cv2.setMouseCallback("ortho_ann", mouse_event)
OVERLAP_PIX = 500
SUB_SHAPE = (2000, 3000)
num_subs_x = math.ceil(full_shape[1] / (P.CNN_INPUT_SHAPE[1] - OVERLAP_PIX))
num_subs_y = math.ceil(full_shape[0] / (P.CNN_INPUT_SHAPE[0] - OVERLAP_PIX))
y_idx = 0
x_idx = 0
save_poly = False
new_sub = False
while True:
    sub_idx_x = int(x_idx*(SUB_SHAPE[1] - OVERLAP_PIX))
    sub_idx_y = int(y_idx*(SUB_SHAPE[0] - OVERLAP_PIX))
    ortho_sub_img = ortho_full[sub_idx_y:min(sub_idx_y+SUB_SHAPE[0], full_shape[0]),
              sub_idx_x:min(sub_idx_x+SUB_SHAPE[1], full_shape[1]), :]
    dem_sub_img = dem_full[sub_idx_y:min(sub_idx_y+SUB_SHAPE[0], full_shape[0]),
                    sub_idx_x:min(sub_idx_x+SUB_SHAPE[1], full_shape[1]), :]

    if ortho_sub_img.shape[0] == 0 or ortho_sub_img.shape[1] == 0 or np.max(ortho_sub_img) == 0:
        if y_idx == (num_subs_y-1):
            exit(0)
        x_idx += 1
    else:
        while True:
            ann_layer = np.zeros_like(ortho_sub_img)
            sml_ortho_ann = np.zeros_like(small_ortho)
            sml_idx_x = sub_idx_x // 10
            sml_idx_y = sub_idx_y // 10
            sml_ortho_ann[sml_idx_y:min(sml_idx_y+SUB_SHAPE[0]//10, full_shape[0]//10), sml_idx_x:min(sml_idx_x+SUB_SHAPE[1]//10, full_shape[1]//10), :] = 20
            for existing_ann_poly in ortho_polygons:
                existing_ann_poly_pix = ((np.array(existing_ann_poly))[:, :2] / P.PIXEL_SCALE).astype(int) - np.array([min_exts[1], min_exts[0]]).astype(int)
                existing_ann_sub_pix = existing_ann_poly_pix - np.array([sub_idx_x, sub_idx_y])
                if len(existing_ann_poly):
                    pnts_sub = existing_ann_sub_pix.reshape((-1, 1, 2))
                    pnts_sml = existing_ann_poly_pix.reshape((-1, 1, 2))
                    try:
                        cv2.polylines(ann_layer, [pnts_sub], False, (255, 255, 255), thickness=3)
                        cv2.polylines(sml_ortho_ann, [pnts_sml//10], False, (255, 255, 255), thickness=1)
                    except:
                        print("err draw line")

            if len(poly_vert_list):
                pnts = np.array(poly_vert_list).reshape((-1, 1, 2))
                cv2.polylines(ann_layer, [pnts], False, (0, 255, 0), thickness=3)

            sml_ortho_display = small_ortho.copy()
            sml_ortho_display[sml_ortho_ann == 20] += 20
            sml_ortho_display[sml_ortho_ann > 20] = 255
            cv2.imshow("Small Ortho", np.vstack([sml_ortho_display, small_dem]))
            ann_display = ortho_sub_img.copy()
            ann_display[np.where(ann_layer > 0)] = 255
            cv2.imshow("ortho_ann", ann_display)
            key = cv2.waitKey(1)

            if key == ord('q'):
                cv2.destroyAllWindows()
                exit(0)
            elif key == ord(' ') or key == ord('b'):
                save_poly = True
            elif key == ord('z'):
                # Delete current ann
                poly_vert_list = []
                ann_layer = np.zeros_like(ortho_sub_img)
            elif key == ord('c'):
                # Show close camera frames
                print("show cams not implemented")
            if key == ord('b'):
                x_idx += 1
                new_sub = True

            if key == 82 or key == ord('w'):
                y_idx = max(y_idx - 1, 0)
                new_sub = True
                save_poly = True
            elif key == 84 or key == ord('s'):
                y_idx = min(y_idx + 1, num_subs_y-1)
                new_sub = True
                save_poly = True
            elif key == 81 or key == ord('a'):
                x_idx = max(x_idx - 1, 0)
                new_sub = True
                save_poly = True
            elif key == 83 or key == ord('d'):
                x_idx = min(x_idx + 1, num_subs_x-1)
                new_sub = True
                save_poly = True

            if save_poly:
                if len(poly_vert_list) > 1:
                    poly_vert_array = np.array(poly_vert_list)
                    elavations = -np.array(dem_sub_img[tuple(poly_vert_array[:, ::-1].transpose().tolist())])
                    poly_verts_wrld_2D = (poly_vert_array + np.array([sub_idx_x+min_exts[1], sub_idx_y+min_exts[0]])) * P.PIXEL_SCALE
                    poly_verts_ortho = np.hstack([poly_verts_wrld_2D, elavations])
                    ortho_polygons.append(poly_verts_ortho.tolist())
                poly_vert_list = []
                save_poly = False
            if new_sub:
                new_sub = False
                break

    if x_idx >= num_subs_x:
        x_idx = 0
        y_idx = min(y_idx + 1, num_subs_y-1)

    ortho_polygons_wrld = []
    for polygon in ortho_polygons:
        polygon_offset = (np.matmul(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), np.array(polygon).transpose()) + ortho_origin[:, None]).transpose()
        ortho_polygons_wrld.append(polygon_offset)

    with open(P.POLY_ANN_LIST_PATH, "wb") as f:
        pickle.dump(ortho_polygons_wrld, f)
    print("Number of Scallops: {}".format(len(ortho_polygons_wrld)))