ROV_DATA = True
INPUT_DATA_PATH = '/home/cosc/research/CVlab/bluerov_data/210113-065012/'

METASHAPE_OUTPUT_DIR = '/local/ScallopReconstructions/lowres_scan_210113_065012/'
METASHAPE_CHKPNT_PATH = METASHAPE_OUTPUT_DIR + 'recon.psx'
POLY_ANN_LIST_PATH = METASHAPE_OUTPUT_DIR + "PolyAnnList_lr.csv"

DATASET_DIR_BASE = "/local/ScallopMaskDataset/"
DATASET_DIR = DATASET_DIR_BASE + "valid_lr/"
MODEL_PATH = "model_final.pth"
CNN_INPUT_SHAPE = (1080, 1920) #(3840, 2160)

INFERENCE_OUTPUT_DIR = "/local/ScallopInferenceOutput/Test/"

TILE_SIZE = 5000
PIXEL_SCALE = 0.0005 # meters per pixel

from detectron2.config import get_cfg
import os

USE_SAVED_MODEL = True
MODEL_USE = 0

cfg = get_cfg()
if not USE_SAVED_MODEL:
    MODEL = "./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    WEIGHT_PATH = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

    if MODEL_USE == 1:
        MODEL = './detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
        WEIGHT_PATH = '../input/global-wheat-detection-model/Detectron_2/faster_rcnn_R_50_FPN_3x.pth'
    elif MODEL_USE == 2:
        MODEL = './detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
        WEIGHT_PATH = '../input/global-wheat-detection-model/Detectron_2_v2/R-50_5k_augmen.pth'
    elif MODEL_USE == 3:
        MODEL = './detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
        WEIGHT_PATH = '../input/global-wheat-detection-model/Detectron_2_v2/R-50_10k_augmen.pth'

    cfg.merge_from_file(MODEL)
    cfg.MODEL.WEIGHTS = WEIGHT_PATH
else:
    cfg.merge_from_file('config.yml')
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, MODEL_PATH)