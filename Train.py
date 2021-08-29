import pathlib as p
import torch, torchvision
import json
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog, transforms
from detectron2.data import build_detection_train_loader
from detectron2.data import DatasetMapper
from detectron2.utils.visualizer import Visualizer
import os
import cv2
import random
import numpy as np
print(torch.__version__)
import Params as P

SHOW_INPUTS = True
TRAIN_SUBDIR = "train_lr"

augs = transforms.AugmentationList([transforms.RandomBrightness(0.5, 1.5),
                                    transforms.RandomRotation([-45, 45]),
                                    transforms.RandomContrast(0.5, 1.5),
                                    transforms.RandomSaturation(0.5, 1.5),
                                    transforms.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                                    transforms.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                                    transforms.RandomExtent(scale_range=(0.1, 2), shift_range=(0.5, 0.5)),
                                    transforms.Resize(P.CNN_INPUT_SHAPE)])

class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augs)
        return build_detection_train_loader(cfg, mapper=mapper)

for d in [TRAIN_SUBDIR]: #, "valid"
    with open(P.DATASET_DIR + d + "/labels.json", 'r') as fp:
        dataset_dicts = json.load(fp)
    DatasetCatalog.register(P.DATASET_DIR + d, lambda d=d: dataset_dicts)
    MetadataCatalog.get(P.DATASET_DIR + d).set(thing_classes=["scallop"])
scallop_metadata = MetadataCatalog.get(P.DATASET_DIR + TRAIN_SUBDIR)

if SHOW_INPUTS:
    for d in random.sample(dataset_dicts, 100):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=scallop_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        img = vis.get_image()[:, :, ::-1]
        cv2.imshow("Original image", img)
        input = transforms.AugInput(img)
        transform = augs(input)
        image_transformed = input.image
        cv2.imshow("Aug image", image_transformed)
        if cv2.waitKey() == ord('q'):
            exit(0)

cfg = P.cfg
cfg.DATASETS.TRAIN = (P.DATASET_DIR + "train",)
cfg.DATASETS.TEST = () #"/local/ScallopMaskDataset/valid"
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 400
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
#cfg.MODEL.BACKBONE.FREEZE_AT = 2
print(cfg)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

f = open('config.yml', 'w')
f.write(cfg.dump())
f.close()