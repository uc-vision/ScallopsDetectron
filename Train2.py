import pathlib as p
import json
import trainer as T
from detectron2.data import DatasetCatalog, transforms
import Params as P
from detectron2.config import get_cfg
import os
from detectron2.utils.logger import logging
from detectron2.data import (
    MetadataCatalog,
)
from detectron2.modeling import build_model
logger = logging.getLogger("detectron2")
import augmentations as A

SHOW_INPUTS = True

VALID_GRAD_SAMPLES = 10
VALID_GRAD_CUTOFF = -0.001 # Per Eval period

try:
    os.mkdir(P.cfg.OUTPUT_DIR)
except OSError as error:
    print(error)

def getDatasetDict(dataset_dir):
    with open(dataset_dir + "/labels.json", 'r') as fp:
        dataset_dict = json.load(fp)
        for data_entry in dataset_dict:
            data_entry["file_name"] = dataset_dir + '/' + data_entry["file_name"].split('/')[-1]
    return dataset_dict

CNN_INPUT_SHAPE = (800, 1333)
augs = [transforms.RandomBrightness(0.8, 1.2),
        transforms.RandomContrast(0.8, 1.2),
        transforms.RandomSaturation(0.8, 1.2),
        transforms.RandomLighting(2),
        transforms.RandomRotation([-90, 0, 90, 180], sample_style="choice"),
        transforms.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        transforms.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        transforms.ResizeScale(min_scale=1, max_scale=4.0, target_height=CNN_INPUT_SHAPE[0], target_width=CNN_INPUT_SHAPE[1]),
        transforms.RandomCrop(crop_type="absolute", crop_size=CNN_INPUT_SHAPE),
        ]
augs = [transforms.RandomCrop(crop_type="absolute", crop_size=CNN_INPUT_SHAPE),
        A.RandomErasing(),
        A.RandomColourNoise(),
        ]
# augs = [transforms.RandomBrightness(0.8, 1.2),
#         transforms.RandomContrast(0.8, 1.2),
#         transforms.RandomSaturation(0.8, 1.2),
#         transforms.RandomLighting(2),
#         transforms.RandomRotation([-45, 45]),
#         transforms.RandomFlip(prob=0.5, horizontal=True, vertical=False),
#         transforms.RandomFlip(prob=0.5, horizontal=False, vertical=True),
#         transforms.RandomExtent(scale_range=(0.7, 1.3), shift_range=(0.3, 0.3)),
#         transforms.Resize(CNN_INPUT_SHAPE)]

datasets = ["gopro_116_0_prop", "gopro_116_0_ortho"]
datasets = [P.DATASET_DIR_BASE+subdir for subdir in datasets]
DatasetCatalog.clear()
for idx, dataset_dir in enumerate(datasets):
    DatasetCatalog.register(dataset_dir, lambda dataset_dir=dataset_dir: getDatasetDict(dataset_dir))
    MetadataCatalog.get(dataset_dir).set(thing_classes=["scallop"])

cfg = get_cfg()
cfg.merge_from_file("./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
CNN_INPUT_SHAPE = (1080, 1920)
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
cfg.OUTPUT_DIR = '/local/trialOut/'
cfg.DATASETS.TRAIN = (datasets[0],)
cfg.DATASETS.TEST = (datasets[1],)
cfg.TEST.EVAL_PERIOD = 100
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 100000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.BACKBONE.FREEZE_AT = 0

model = build_model(cfg)
T.do_train(cfg, model, augs, resume=False)

# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# model = build_model(cfg)
# do_train(cfg, model, resume=P.USE_SAVED_MODEL)
#
# f = open(cfg.OUTPUT_DIR+'/config.yml', 'w')
# f.write(cfg.dump())
# f.close()