from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog, transforms
import cv2
import os
import pathlib
import Params as P
import json

DS_DIV = 2
TEST_IMG_DIR = "/local/ScallopMaskDataset/train/"#"/home/cosc/research/CVlab/bluerov_data/210113-065012/"#

paths = pathlib.Path(TEST_IMG_DIR)
if any(fname.endswith('.pgm') for fname in os.listdir(TEST_IMG_DIR)):
    img_fns = [str(fn) for fn in paths.iterdir() if str(fn)[-4:] == '.pgm']
    BAYER = True
else:
    img_fns = [str(fn) for fn in paths.iterdir() if str(fn)[-4:] == '.jpg']
    BAYER = False
print("Num images in dir: {}".format(len(img_fns)))
img_fns.sort()

if any(fname.endswith('.json') for fname in os.listdir(TEST_IMG_DIR)):
    with open(TEST_IMG_DIR + "/labels.json", 'r') as fp:
        dataset_dicts = json.load(fp)
    DatasetCatalog.register(TEST_IMG_DIR, lambda d='eval': dataset_dicts)
    MetadataCatalog.get(TEST_IMG_DIR).set(thing_classes=["scallop"])
scallop_metadata = MetadataCatalog.get(TEST_IMG_DIR)

cfg = P.cfg
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
#cfg.DATASETS.TEST = ("/local/ScallopMaskDataset/val", )
predictor = DefaultPredictor(cfg)

for path in img_fns:
    im = cv2.imread(path)
    if BAYER:
        im = cv2.cvtColor(im[:, :, 0][:, :, None], cv2.COLOR_BAYER_BG2BGR)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    out_image = v.get_image()[:, :, ::-1]
    cv2.imshow("Detectron input", cv2.resize(im, (im.shape[1]//DS_DIV, im.shape[0]//DS_DIV)))
    cv2.imshow("Detectron output", cv2.resize(out_image, (out_image.shape[1]//DS_DIV, out_image.shape[0]//DS_DIV)))
    if cv2.waitKey() == ord('q'):
        break
