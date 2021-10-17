import pathlib as p
import torch, torchvision
import json
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog, transforms
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import DatasetMapper
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.utils.visualizer import Visualizer
import os
import cv2
import random
import numpy as np
print(torch.__version__)
import Params as P
import DetectronLossHooks as DLH

SHOW_INPUTS = False
TRAIN_SUBDIR = "train_lr"
VAL_SUBDIR = "valid_lr"

if not P.USE_SAVED_MODEL:
    [path.unlink() for path in p.Path(P.cfg.OUTPUT_DIR).iterdir()]

augs = [transforms.RandomBrightness(0.8, 1.2),
        transforms.RandomRotation([-45, 45]),
        transforms.RandomContrast(0.8, 1.2),
        transforms.RandomSaturation(0.8, 1.2),
        transforms.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        transforms.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        transforms.RandomExtent(scale_range=(0.7, 1.3), shift_range=(0.3, 0.3)),
        transforms.Resize(P.CNN_INPUT_SHAPE)]

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augs)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = DatasetMapper(cfg, is_train=True)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    # @classmethod
    # def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    #     if output_folder is None:
    #         output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    #     evaluator_list = []
    #     # evaluator_list.append(
    #     #     SemSegEvaluator(
    #     #         dataset_name,
    #     #         distributed=True,
    #     #         num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
    #     #         ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
    #     #         output_dir=output_folder,
    #     #     ))
    #     return DatasetEvaluators(evaluator_list)

    def build_hooks(self):
        hooks = super(MyTrainer, self).build_hooks()
        cfg = self.cfg
        if len(cfg.DATASETS.TEST) > 0:
            loss_eval_hook = DLH.LossEvalHook(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                MyTrainer.build_test_loader(cfg, cfg.DATASETS.TEST[0]),
            )
            hooks.insert(-1, loss_eval_hook)
        return hooks


for d in [TRAIN_SUBDIR, VAL_SUBDIR]:
    with open(P.DATASET_DIR_BASE + d + "/labels.json", 'r') as fp:
        dataset_dicts = json.load(fp)
    DatasetCatalog.register(P.DATASET_DIR_BASE + d, lambda d=d: dataset_dicts)
    MetadataCatalog.get(P.DATASET_DIR_BASE + d).set(thing_classes=["scallop"])

scallop_metadata = MetadataCatalog.get(P.DATASET_DIR_BASE + TRAIN_SUBDIR)
if SHOW_INPUTS:
    for d in random.sample(dataset_dicts, 100):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=scallop_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        img = vis.get_image()[:, :, ::-1]
        cv2.imshow("Original image", img)
        input = transforms.AugInput(img)
        transform = transforms.AugmentationList(augs)(input)
        image_transformed = input.image
        cv2.imshow("Aug image", image_transformed)
        if cv2.waitKey() == ord('q'):
            exit(0)

cfg = P.cfg
cfg.DATASETS.TRAIN = (P.DATASET_DIR_BASE + TRAIN_SUBDIR,)
cfg.DATASETS.TEST = (P.DATASET_DIR_BASE + VAL_SUBDIR,)
cfg.TEST.EVAL_PERIOD = 10
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 8000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
#cfg.MODEL.BACKBONE.FREEZE_AT = 2
print(cfg)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=P.USE_SAVED_MODEL)
trainer.train()

f = open('config.yml', 'w')
f.write(cfg.dump())
f.close()