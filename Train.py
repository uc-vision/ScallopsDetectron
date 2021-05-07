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

USE_SAVED_MODEL = True

DATASET_DIR = "/local/ScallopMaskDataset/"

augs = transforms.AugmentationList([transforms.RandomBrightness(0.5, 2),
        transforms.RandomContrast(0.5, 2),
        transforms.RandomSaturation(0.5, 2),
        transforms.RandomFlip(prob=0.5),
        transforms.RandomExtent(scale_range=(0.1, 2), shift_range=(1, 1))])
class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augs)
        return build_detection_train_loader(cfg, mapper=mapper)


for d in ["train", "valid"]:
    with open(DATASET_DIR + d + "/labels.json", 'r') as fp:
        dataset_dicts = json.load(fp)
    DatasetCatalog.register(DATASET_DIR + d, lambda d=d: dataset_dicts)
    MetadataCatalog.get(DATASET_DIR + d).set(thing_classes=["scallop"])
scallop_metadata = MetadataCatalog.get(DATASET_DIR + "train")

cfg = get_cfg()
if USE_SAVED_MODEL:
    cfg.merge_from_file('config.yml')
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
else:
    cfg.merge_from_file("./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

if True:
    for d in random.sample(dataset_dicts, 100):
        img = cv2.imread(d["file_name"])
        input = transforms.AugInput(img)
        transform = augs(input)
        image_transformed = input.image

        visualizer = Visualizer(img[:, :, ::-1], metadata=scallop_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        img = vis.get_image()[:, :, ::-1]
        print(img.shape)
        cv2.imshow("Original image", img)

        visualizer = Visualizer(image_transformed[:, :, ::-1], metadata=scallop_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        image_transformed = vis.get_image()[:, :, ::-1]
        print(image_transformed.shape)
        cv2.imshow("Aug image", image_transformed)
        cv2.waitKey()

cfg.DATASETS.TRAIN = ("/local/ScallopMaskDataset/train",)
cfg.DATASETS.TEST = () #"/local/ScallopMaskDataset/valid"
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
print(cfg)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

f = open('config.yml', 'w')
f.write(cfg.dump())
f.close()