import numpy as np
import os
import pathlib
from detectron2.data import transforms
import gc
from detectron2.engine import launch
from utils import maskrcnn_setup, train_net, augmentations as A
from detectron2.data import build_detection_train_loader
import cv2
from datetime import datetime


WRITE = True
RESUME = False
SHOW_TRAINING_IMGS = False

BASE_DIR = '/local/'#'/scratch/data/tkr25/'  #
NUM_GPUS = 1
BATCH_SIZE = 8

CNN_INPUT_SHAPE = (800, 1333)


augs = [transforms.RandomBrightness(0.8, 1.2),
        transforms.RandomContrast(0.8, 1.2),
        transforms.RandomSaturation(0.8, 1.2),
        transforms.RandomLighting(2),
        transforms.RandomRotation([-90, 0, 90, 180], sample_style="choice"),
        transforms.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        transforms.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        #transforms.ResizeScale(min_scale=1, max_scale=4.0, target_height=CNN_INPUT_SHAPE[0], target_width=CNN_INPUT_SHAPE[1]),
        transforms.RandomCrop(crop_type="absolute", crop_size=CNN_INPUT_SHAPE),
        #A.RandomErasing(),
        #A.RandomColourNoise(),
        A.GeometricTransform()
        ]
no_augs = [transforms.RandomCrop(crop_type="absolute", crop_size=CNN_INPUT_SHAPE),]


EXP_START_IDX = 0
experiment_titles = ["HR+LR PROP MOREDATA AUGS",
                     ]
augmentation_sets = [augs,
                     ]
valid_datasets = ['lowres_scan_210113_064700_prop', 'gopro_116_0_ortho', 'gopro_116_0_prop']
train_valid_dataset_sets = [[['gopro_115_prop', 'gopro_119_prop', 'lowres_scan_210113_065012_prop'], valid_datasets],
                            ]
train_valid_dataset_sets = [[[BASE_DIR+'ScallopMaskDataset/'+sssd for sssd in ssd] for ssd in sd] for sd in train_valid_dataset_sets]


def main(args):
    cfg = maskrcnn_setup.setup(args)

    if SHOW_TRAINING_IMGS:
        mapper = A.CustomMapper(cfg, is_train=True, augmentations=augmentation_sets[args["IDX"]])
        data_loader = build_detection_train_loader(cfg, mapper=mapper)
        for data in data_loader:
            image = data[0]['image'].to('cpu').numpy().transpose([1, 2, 0])
            print(data[0]['image'].shape)
            cv2.imshow("Training data", image)
            cv2.waitKey()
            print(data)

    trainer = train_net.Trainer(cfg, args["augmentations"])
    trainer.resume_or_load(resume=RESUME)
    return trainer.train()


if __name__ == '__main__':
    for exp_idx in np.arange(EXP_START_IDX, len(experiment_titles)):
        exp_title = experiment_titles[exp_idx]
        exp_augs = augmentation_sets[exp_idx]
        exp_datasets = train_valid_dataset_sets[exp_idx]
        output_dir = BASE_DIR + 'ScallopMaskRCNNOutputs/' + exp_title

        try:
            os.mkdir(output_dir)
        except OSError as error:
            print(error)
        if WRITE and not RESUME:
            [path.unlink() for path in pathlib.Path(output_dir).iterdir()]
            with open(output_dir + "/Info.txt", 'w') as info_f:
                info_f.write("Experiment Title: {}\n".format(exp_title))
                info_f.write("Experiment Date: {:%B %d, %Y}\n\n".format(datetime.now()))
                info_f.write("Training Directories:\n")
                info_f.writelines(dtst + '\n' for dtst in exp_datasets[0])
                info_f.write("\nAugmentations:\n")
                info_f.writelines([str(aug) + '\n' for aug in exp_augs])

        launch(
            main,
            NUM_GPUS,
            num_machines=1,
            machine_rank=0,
            dist_url='tcp://127.0.0.1:5000'+str(np.random.randint(0, 9)),
            args=({"output_dir":output_dir, "dataset_dirs":exp_datasets, "num_gpus":NUM_GPUS, "gpu_batch_size":BATCH_SIZE, "augmentations":exp_augs, "IDX":exp_idx},),
        )
        gc.collect()
