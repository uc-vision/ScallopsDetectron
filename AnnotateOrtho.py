import numpy as np
import pathlib as p
import cv2
from tqdm import tqdm
import Params as P
import pickle

ortho_paths = [str(path) for path in p.Path(P.METASHAPE_OUTPUT_DIR + 'ortho/').iterdir()]
dem_paths = [str(path) for path in p.Path(P.METASHAPE_OUTPUT_DIR + 'dem/').iterdir()]
dem_paths.sort()
ortho_paths.sort()
assert len(ortho_paths) == len(dem_paths)
#TODO: add check that all tile filenames line up

path_tuples = list(zip(ortho_paths, dem_paths))
tile_offsets = np.array([[float(path.split('-')[-2]), -float(path.split('-')[-1][:-4])] for path in ortho_paths])
tile_offsets *= P.TILE_SIZE
ortho_origin = np.load(P.METASHAPE_OUTPUT_DIR + 'ortho_origin.npy')

key = ''
poly_vert_list = []
def draw_ann(event, x, y, flags, param):
    global poly_vert_list
    if event==cv2.EVENT_RBUTTONDOWN:
        poly_vert_list.append([x, y])

wrld_poly_points = []
cv2.namedWindow("ortho_ann", cv2.WINDOW_GUI_NORMAL)
cv2.setMouseCallback("ortho_ann", draw_ann)
for idx, (ortho_path, dem_path) in tqdm(enumerate(path_tuples)):
    ortho_tile = cv2.imread(ortho_path)
    dem_tile = cv2.imread(dem_path, cv2.IMREAD_ANYDEPTH)
    ann_layer = np.zeros_like(ortho_tile)

    while True:
        if len(poly_vert_list):
            pnts = np.array(poly_vert_list).reshape((-1, 1, 2))
            cv2.polylines(ann_layer, [pnts], False, (0, 255, 0), thickness=3)
        display = ortho_tile.copy()
        display[np.where(ann_layer > 0)] = 255
        cv2.imshow("ortho_ann", display)
        key = cv2.waitKey(1)

        if key == ord('q'):
            cv2.destroyAllWindows()
            exit(0)
        elif key == ord(' ') or key == ord('b'):
            if len(poly_vert_list) > 1:
                poly_vert_array = np.array(poly_vert_list)
                elavations = np.array(dem_tile[tuple(poly_vert_array[:, ::-1].transpose().tolist())])[:, None]
                poly_vert_array[:, 1] *= -1
                poly_verts_wrld_2D = (poly_vert_array + tile_offsets[idx]) * P.PIXEL_SCALE
                poly_verts_wrld = np.hstack([poly_verts_wrld_2D, elavations]) + ortho_origin
                wrld_poly_points.append(poly_verts_wrld.tolist())
            ortho_tile[np.where(ann_layer[:, :, 1] > 0)] = (255, 255, 255)
            poly_vert_list = []
        elif key == ord('z'):
            # Delete current ann
            poly_vert_list = []
            ann_layer = np.zeros_like(ortho_tile)
        if key == ord('b'):
            break

with open(P.POLY_ANN_LIST_FN, "wb") as f:
    pickle.dump(wrld_poly_points, f)
cv2.destroyAllWindows()
