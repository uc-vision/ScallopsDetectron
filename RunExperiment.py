import numpy as np
import os
import pathlib
from detectron2.data import transforms
import time
import gc
from detectron2.engine import launch
import train_net

WRITE = True
BASE_DIR = '/scratch/data/tkr25/'  #'/local/'  #
NUM_GPUS = 4
BATCH_SIZE = 2

CNN_INPUT_SHAPE = (800, 1333)

experiment_titles = ["Label propagation NOAUGS", "ortho NOAUGS", "ortho AUGS", "Label propagation AUGS", "HR label prop data AUGS", "HR+LR PROP AUGS", "test"]
augs = [transforms.RandomBrightness(0.8, 1.2),
        transforms.RandomContrast(0.8, 1.2),
        transforms.RandomSaturation(0.8, 1.2),
        transforms.RandomLighting(2),
        transforms.RandomRotation([-45, 45]),
        transforms.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        transforms.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        transforms.ResizeScale(min_scale=1, max_scale=8.0, target_height=CNN_INPUT_SHAPE[0], target_width=CNN_INPUT_SHAPE[1]),
        transforms.RandomCrop(crop_type="absolute", crop_size=CNN_INPUT_SHAPE),
        ]
#transforms.RandomExtent(scale_range=(0.7, 1.3), shift_range=(0.3, 0.3)),

augmentation_sets = [[], [], augs, augs, augs, augs, []]
train_valid_dataset_sets = [[['ScallopMaskDataset/train_lr'], ['ScallopMaskDataset/valid_lr']],
                            [['ScallopMaskDataset/train_lr_ortho'], ['ScallopMaskDataset/valid_lr_ortho']],
                            [['ScallopMaskDataset/train_lr_ortho'], ['ScallopMaskDataset/valid_lr_ortho']],
                            [['ScallopMaskDataset/train_lr'], ['ScallopMaskDataset/valid_lr']],
                            [['ScallopMaskDataset/train_hr'], ['ScallopMaskDataset/valid_lr_ortho', 'ScallopMaskDataset/train_lr_ortho']], #'ScallopMaskDataset/valid_lr',
                            [['ScallopMaskDataset/train_hr'], ['ScallopMaskDataset/valid_lr_ortho', 'ScallopMaskDataset/train_lr_ortho']],
                            ]
train_valid_dataset_sets = [[[BASE_DIR+sssd for sssd in ssd] for ssd in sd] for sd in train_valid_dataset_sets]


def main(args):
    cfg = train_net.setup(args)
    trainer = train_net.Trainer(cfg, args["augmentations"])
    trainer.resume_or_load(resume=False)
    return trainer.train()


if __name__ == '__main__':
    start_idx = 4
    for exp_idx in np.arange(start_idx, len(experiment_titles)):
        exp_title = experiment_titles[exp_idx]
        exp_augs = augmentation_sets[exp_idx]
        exp_datasets = train_valid_dataset_sets[exp_idx]
        output_dir = BASE_DIR + 'ScallopMaskRCNNOutputs/' + exp_title
        try:
            os.mkdir(output_dir)
        except OSError as error:
            print(error)
        if WRITE:
            [path.unlink() for path in pathlib.Path(output_dir).iterdir()]

        launch(
            main,
            NUM_GPUS,
            num_machines=1,
            machine_rank=0,
            dist_url='tcp://127.0.0.1:5000'+str(np.random.randint(0, 9)),
            args=({"output_dir":output_dir, "dataset_dirs":exp_datasets, "num_gpus":NUM_GPUS, "gpu_batch_size":BATCH_SIZE, "augmentations":exp_augs},),
        )

        gc.collect()
