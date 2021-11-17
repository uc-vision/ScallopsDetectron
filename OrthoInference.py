import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from tqdm import tqdm
import pathlib
import Params as P
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

SHOW = True
INFERENCE_ORTHO_PATH = P.METASHAPE_OUTPUT_DIR + "ortho/"
EDGE_LIMIT_PIX = int(0.05 / P.PIXEL_SCALE)  # How close a detection can be to the border to be counted
OVERLAP_PIX = 2*EDGE_LIMIT_PIX

cfg = P.cfg
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.92
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
cfg.DATASETS.TEST = ("/local/ScallopMaskDataset/val", )
predictor = DefaultPredictor(cfg)

ortho_paths = [str(path) for path in pathlib.Path(P.METASHAPE_OUTPUT_DIR + 'ortho/').iterdir()]
dem_paths = [str(path) for path in pathlib.Path(P.METASHAPE_OUTPUT_DIR + 'dem/').iterdir()]
dem_paths.sort()
ortho_paths.sort()
assert len(ortho_paths) == len(dem_paths)
ortho_origin = np.load(P.METASHAPE_OUTPUT_DIR + 'ortho_origin.npy')


def LoadTiles(paths, dims=3, dtype=np.uint8, read_flags=None):
    tile_offsets_rc = np.array([[float(path.split('-')[-1][:-4]), float(path.split('-')[-2])] for path in paths])
    tile_offsets_rc *= P.TILE_SIZE
    tile_extents = tile_offsets_rc + P.TILE_SIZE
    max_extents = np.max(tile_extents, axis=0).astype(int)
    ortho_full = np.zeros((max_extents[0], max_extents[1], dims), dtype=dtype)
    full_shape = ortho_full.shape
    print("Ortho Shape: {}, loading tiles...".format(full_shape))
    for ortho_tile_path, offset in tqdm(list(zip(paths, tile_offsets_rc.astype(int)))):
        ortho_tile = cv2.imread(ortho_tile_path, read_flags)
        tile_shape = ortho_tile.shape
        ortho_full[offset[0]:(offset[0]+tile_shape[0]), offset[1]:(offset[1]+tile_shape[1])] = ortho_tile.reshape((tile_shape[0], tile_shape[1], -1))
    return ortho_full, full_shape


ortho_full, full_shape = LoadTiles(ortho_paths)
dem_full, shape = LoadTiles(dem_paths, dims=1, read_flags=cv2.IMREAD_ANYDEPTH, dtype=np.float32)
RSZ_MOD = 5
small_ortho = cv2.resize(ortho_full, (full_shape[1]//RSZ_MOD, full_shape[0]//RSZ_MOD))
if SHOW:
    cv2.namedWindow("ortho full", cv2.WINDOW_NORMAL)

print("Running Inference...")
scallop_detections = []
num_subs_x = math.ceil(full_shape[1] / (P.CNN_INPUT_SHAPE[1] - OVERLAP_PIX))
num_subs_y = math.ceil(full_shape[0] / (P.CNN_INPUT_SHAPE[0] - OVERLAP_PIX))
for y_idx in tqdm(range(num_subs_y)):
    for x_idx in range(num_subs_x):
        sub_idx_x = int(x_idx*(P.CNN_INPUT_SHAPE[1] - OVERLAP_PIX))
        sub_idx_y = int(y_idx*(P.CNN_INPUT_SHAPE[0] - OVERLAP_PIX))
        sub_img = ortho_full[sub_idx_y:min(sub_idx_y+P.CNN_INPUT_SHAPE[0], full_shape[0]),
                  sub_idx_x:min(sub_idx_x+P.CNN_INPUT_SHAPE[1], full_shape[1]), :]
        sub_shape = sub_img.shape
        edge_box = (EDGE_LIMIT_PIX, EDGE_LIMIT_PIX, sub_shape[1]-EDGE_LIMIT_PIX, sub_shape[0]-EDGE_LIMIT_PIX)

        outputs = predictor(sub_img)
        v = Visualizer(sub_img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
        instances = outputs["instances"].to("cpu")
        v = v.draw_instance_predictions(instances)
        out_image = v.get_image()[:, :, ::-1].copy()

        masks = instances._fields['pred_masks']
        bboxes = instances._fields['pred_boxes']
        scores = instances._fields['scores']
        for mask, box, score in list(zip(masks, bboxes, scores)):
            mask_pnts = np.array(np.where(mask))[::-1].transpose()
            (x, y), radius = cv2.minEnclosingCircle(mask_pnts)
            if edge_box[0] < x <= edge_box[2] and edge_box[1] < y <= edge_box[3]:
                cv2.circle(out_image, (int(x), int(y)), int(radius), color=(0, 255, 0), thickness=2)
                cv2.circle(small_ortho, ((sub_idx_x + int(x))//RSZ_MOD, (sub_idx_y + int(y))//RSZ_MOD),
                           int(radius)//RSZ_MOD, color=(0, 255, 0), thickness=1)
                cv2.putText(small_ortho, str(round(score.numpy().item(), 2)),
                            ((sub_idx_x + int(x) - 60)//RSZ_MOD, (sub_idx_y + int(y) + 20)//RSZ_MOD),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                bbox_np = box.numpy()
                local_pixpos = np.array((bbox_np[2]/2 + bbox_np[0]/2, (bbox_np[3]/2 + bbox_np[1]/2)))
                global_pixpos = local_pixpos + (sub_idx_x, sub_idx_y)

                mask_pnts_sub = mask_pnts[::10, :]
                mask_pnt_array = np.repeat(mask_pnts_sub[:, None, :], mask_pnts_sub.shape[0], axis=1)
                mask_pnt_dists = np.linalg.norm(mask_pnt_array - mask_pnt_array.transpose((1, 0, 2)), axis=2)
                max_dist_idxs = np.unravel_index(mask_pnt_dists.argmax(), mask_pnt_dists.shape)
                extrema_pnts = mask_pnts_sub[max_dist_idxs, :].transpose() #[[x1, x2], [y1, y2]]
                global_pnts = extrema_pnts + [2*[sub_idx_x], 2*[sub_idx_y]]
                pnt_elavations = dem_full[[int(global_pnts[1, 0]), int(global_pnts[1, 1])], [int(global_pnts[0, 0]), int(global_pnts[0, 1])]]
                pnts_3D = np.vstack([global_pnts*P.PIXEL_SCALE, pnt_elavations.transpose()])
                size_3D = np.clip(np.linalg.norm(pnts_3D[:, 0] - pnts_3D[:, 1]), 0.01, 0.15)

                cv2.circle(out_image, (int(extrema_pnts[0, 0]), int(extrema_pnts[1, 0])), 10, color=(0, 0, 255), thickness=-1)
                cv2.circle(out_image, (int(extrema_pnts[0, 1]), int(extrema_pnts[1, 1])), 10, color=(0, 0, 255), thickness=-1)
                cv2.circle(small_ortho, (int(global_pnts[0, 0])//RSZ_MOD, int(global_pnts[1, 0])//RSZ_MOD), 2, color=(0, 0, 255), thickness=-1)
                cv2.circle(small_ortho, (int(global_pnts[0, 1])//RSZ_MOD, int(global_pnts[1, 1])//RSZ_MOD), 2, color=(0, 0, 255), thickness=-1)
                # cv2.imshow("test", out_image)
                # cv2.imshow("test2", small_ortho)
                # cv2.waitKey()

                #TODO: scallopness thresholding?

                scallop_detections.append((global_pixpos, size_3D, score.numpy()))

        if SHOW:
            cv2.rectangle(out_image, edge_box[:2], edge_box[2:], (0, 0, 255), thickness=1)
            cv2.rectangle(small_ortho, (sub_idx_x//RSZ_MOD, sub_idx_y//RSZ_MOD),
                          ((sub_idx_x+sub_shape[1])//RSZ_MOD, (sub_idx_y+sub_shape[0])//RSZ_MOD),
                          color=(255, 255, 255), thickness=1)
            cv2.rectangle(small_ortho, ((sub_idx_x+EDGE_LIMIT_PIX)//RSZ_MOD, (sub_idx_y+EDGE_LIMIT_PIX)//RSZ_MOD),
                          ((sub_idx_x+sub_shape[1]-EDGE_LIMIT_PIX)//RSZ_MOD, (sub_idx_y+sub_shape[0]-EDGE_LIMIT_PIX)//RSZ_MOD),
                          color=(0, 0, 255), thickness=1)
            cv2.imshow("ortho full", small_ortho)
            CNN_OUTPUT_RSZ_MOD = 1
            cv2.imshow("Labelled sub image", cv2.resize(out_image, (out_image.shape[1]//CNN_OUTPUT_RSZ_MOD, out_image.shape[0]//CNN_OUTPUT_RSZ_MOD)))
            key = cv2.waitKey(1)
            if key == ord('q'):
                exit(0)

scallop_pnts = np.array([loc for loc, ssize, conf in scallop_detections])
#scallop_pnts[:, 1] = np.max(scallop_pnts[:, 1]) - scallop_pnts[:, 1]
scallop_pnts = (scallop_pnts * P.PIXEL_SCALE + ortho_origin[:2])
scallop_sizes = np.array([ssize for loc, ssize, conf in scallop_detections])
plt.figure(1)
plt.title("Scallop Spatial Distribution [m]")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.plot(scallop_pnts[:, 0], scallop_pnts[:, 1], 'ro')
plt.grid(True)
plt.savefig(P.INFERENCE_OUTPUT_DIR + "ScallopSpatialDistOrtho.jpeg")
plt.figure(2)
plt.title("Scallop Size Distribution (freq. vs size [m])")
plt.ylabel("Frequency")
plt.xlabel("Scallop Width [m]")
plt.hist(scallop_sizes, bins=100)
plt.figtext(0.15, 0.85, "Total count: {}".format(scallop_pnts.shape[0]))
plt.grid(True)
plt.savefig(P.INFERENCE_OUTPUT_DIR + "ScallopSizeDistOrtho.jpeg")
plt.show()

cv2.imwrite(P.INFERENCE_OUTPUT_DIR + "OrthoSmlLabelledOrtho.jpeg", small_ortho)

if SHOW:
    cv2.imshow("ortho full", small_ortho)
    cv2.waitKey()