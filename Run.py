# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import os
import pathlib as P

TEST_IMG_DIR = "/home/cosc/research/CVlab/bluerov_data/210113-064700/"#"/local/ScallopMaskDataset/test/"
BAYER = True

paths = P.Path(TEST_IMG_DIR)
img_fns = [str(fn) for fn in paths.iterdir()]

cfg = get_cfg()
cfg.merge_from_file('config.yml')
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
cfg.DATASETS.TEST = ("/local/ScallopMaskDataset/val", )
predictor = DefaultPredictor(cfg)
DS_DIV = 1

for path in img_fns:
    im = cv2.imread(path)
    if BAYER:
        im = cv2.cvtColor(im[:, :, 0][:, :, None], cv2.COLOR_BAYER_BG2BGR)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    out_image = v.get_image()[:, :, ::-1]
    cv2.imshow("Detectron input", cv2.resize(im, (im.shape[1]//DS_DIV, im.shape[0]//DS_DIV)))
    cv2.imshow("Detectron output", cv2.resize(out_image, (out_image.shape[1]//DS_DIV, out_image.shape[0]//DS_DIV)))
    if cv2.waitKey() == ord('q'):
        break
