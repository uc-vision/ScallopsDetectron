ROV_DATA = False
ROV_DATA_PATH = '/home/cosc/research/CVlab/bluerov_data/210113-065012/'

# RECON / ANN WORKING DIRECTORY
METASHAPE_OUTPUT_DIR = '/local/ScallopReconstructions/lowres_scan_210113_064700/' #lowres_scan_210114_023202/' #'/local/ScallopReconstructions/gopro_115/'#
METASHAPE_CHKPNT_PATH = METASHAPE_OUTPUT_DIR + 'recon.psx'
POLY_ANN_LIST_PATH = METASHAPE_OUTPUT_DIR + "PolyAnnList_lr.csv"
VALID_ORTHO_POINTS_PATH = METASHAPE_OUTPUT_DIR + "valid_ortho_points.npy"

ORTHOSUB_OVERLAP_PIX = 500
ORTHOSUB_SHAPE = (1500, 3000)

DATASET_DIR_BASE = "/local/ScallopMaskDataset/"
# DATASET OUTPUT DIRECTORY
DATASET_DIR = DATASET_DIR_BASE + "valid_lr/"

MODEL_PATH = "model_final.pth"

TILE_SIZE = 5000
PIXEL_SCALE = 0.0005  # meters per pixel

from detectron2.config import get_cfg
import os

USE_SAVED_MODEL = False

cfg = get_cfg()
if not USE_SAVED_MODEL:
    MODEL = "./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    WEIGHT_PATH = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    cfg.merge_from_file(MODEL)
    cfg.MODEL.WEIGHTS = WEIGHT_PATH
else:
    cfg.merge_from_file('/local/ScallopMaskRCNNOutputs/train_lr_ortho_NOaug/'+'config.yml')
    cfg.MODEL.WEIGHTS = os.path.join('/local/ScallopMaskRCNNOutputs/train_lr_ortho_NOaug/', MODEL_PATH)