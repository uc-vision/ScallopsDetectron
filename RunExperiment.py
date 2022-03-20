import numpy as np
import os
import pathlib
from detectron2.data import transforms
import time
import gc
from detectron2.engine import launch
import train_net
import augmentations as A
from detectron2.data import DatasetCatalog
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data import build_detection_train_loader
import json
import cv2

WRITE = True
RESUME = False
SHOW_TRAINING_IMGS = False

BASE_DIR = '/scratch/data/tkr25/'  #'/local/'#
NUM_GPUS = 4
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
        A.RandomErasing(),
        A.RandomColourNoise(),
        ]
augs_bw = [transforms.RandomBrightness(0.8, 1.2),
          transforms.RandomContrast(0.8, 1.2),
          transforms.RandomSaturation(0.8, 1.2),
          transforms.RandomLighting(2),
          transforms.RandomRotation([-90, 0, 90, 180], sample_style="choice"),
          transforms.RandomFlip(prob=0.5, horizontal=True, vertical=False),
          transforms.RandomFlip(prob=0.5, horizontal=False, vertical=True),
          transforms.ResizeScale(min_scale=1, max_scale=4.0, target_height=CNN_INPUT_SHAPE[0], target_width=CNN_INPUT_SHAPE[1]),
          transforms.RandomCrop(crop_type="absolute", crop_size=CNN_INPUT_SHAPE),
          A.RandomErasing(),
          ]
no_augs = [transforms.RandomCrop(crop_type="absolute", crop_size=CNN_INPUT_SHAPE),]

START_IDX = 0
experiment_titles = ["HR PROP BW AUGS",
                     "HR PROP BW NOAUGS",
                     "HR ORTHO BW AUGS",
                     "HR PROP+ORTHO BW AUGS",
                     ]
augmentation_sets = [augs_bw, no_augs, augs_bw, augs_bw]
valid_datasets = ['gopro_116_0_ortho_bw', 'gopro_116_0_prop_bw', 'lowres_scan_210113_064700_prop', 'gopro_116_0_ortho', 'gopro_116_0_prop']
train_valid_dataset_sets = [[['gopro_115_prop_bw'], valid_datasets],
                            [['gopro_115_prop_bw'], valid_datasets],
                            [['gopro_115_ortho_bw'], valid_datasets],
                            [['gopro_115_prop_bw', 'gopro_115_ortho_bw'], valid_datasets],
                            ]
train_valid_dataset_sets = [[[BASE_DIR+'ScallopMaskDataset/'+sssd for sssd in ssd] for ssd in sd] for sd in train_valid_dataset_sets]


def getDatasetDict(dataset_dir):
    with open(dataset_dir + "/labels.json", 'r') as fp:
        dataset_dict = json.load(fp)
        for data_entry in dataset_dict:
            data_entry["file_name"] = dataset_dir + '/' + data_entry["file_name"].split('/')[-1]
    return dataset_dict


def setup(args):
    train_dirs, valid_dirs = args["dataset_dirs"]
    DatasetCatalog.clear()
    for idx, dataset_dir in enumerate(train_dirs+valid_dirs):
        DatasetCatalog.register(dataset_dir, lambda dataset_dir=dataset_dir: getDatasetDict(dataset_dir))
        MetadataCatalog.get(dataset_dir).set(thing_classes=["scallop"])

    cfg = get_cfg()
    cfg.merge_from_file("./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

    cfg.NUM_GPUS = args["num_gpus"]
    cfg.SOLVER.IMS_PER_BATCH = args["gpu_batch_size"] * cfg.NUM_GPUS
    cfg.SOLVER.REFERENCE_WORLD_SIZE = cfg.NUM_GPUS
    cfg.SOLVER.WARMUP_ITERS = cfg.SOLVER.WARMUP_ITERS // cfg.NUM_GPUS

    cfg.OUTPUT_DIR = args["output_dir"]
    cfg.DATASETS.TRAIN = train_dirs
    cfg.DATASETS.TEST = valid_dirs
    cfg.TEST.EVAL_PERIOD = 500 // cfg.NUM_GPUS
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000 // cfg.NUM_GPUS

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.0

    cfg.SOLVER.BASE_LR = cfg.NUM_GPUS * 0.001
    cfg.SOLVER.GAMMA = 0.1
    # The iteration number to decrease learning rate by GAMMA.
    cfg.SOLVER.STEPS = (5000 // cfg.NUM_GPUS,)

    cfg.SOLVER.MAX_ITER = 20000 // cfg.NUM_GPUS
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    # The weight decay that's applied to parameters of normalization layers
    # (typically the affine transformation)
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0

    cfg.MODEL.BACKBONE.FREEZE_AT = 0

    # Default from ImageNet dataset: [103.530, 116.280, 123.675]
    cfg.MODEL.PIXEL_MEAN = [100, 100,  70]
    # When using pre-trained models in Detectron1 or any MSRA models,
    # std has been absorbed into its conv1 weights, so the std needs to be set 1.
    # Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]

    cfg.INPUT.RANDOM_FLIP = "none"
    # `True` if cropping is used for data augmentation during training
    cfg.INPUT.CROP.ENABLED = False
    # Cropping type. See documentation of `detectron2.data.transforms.RandomCrop` for explanation.
    cfg.INPUT.CROP.TYPE = "relative_range"
    # Size of crop in range (0, 1] if CROP.TYPE is "relative" or "relative_range" and in number of
    # pixels if CROP.TYPE is "absolute"
    cfg.INPUT.CROP.SIZE = [0.9, 0.9]

    # ---------------------------------------------------------------------------- #
    # Anchor generator options
    # ---------------------------------------------------------------------------- #
    # Anchor sizes (i.e. sqrt of area) in absolute pixels w.r.t. the network input.
    # Format: list[list[float]]. SIZES[i] specifies the list of sizes to use for
    # IN_FEATURES[i]; len(SIZES) must be equal to len(IN_FEATURES) or 1.
    # When len(SIZES) == 1, SIZES[0] is used for all IN_FEATURES.
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
    # Anchor aspect ratios. For each area given in `SIZES`, anchors with different aspect
    # ratios are generated by an anchor generator.
    # Format: list[list[float]]. ASPECT_RATIOS[i] specifies the list of aspect ratios (H/W)
    # to use for IN_FEATURES[i]; len(ASPECT_RATIOS) == len(IN_FEATURES) must be true,
    # or len(ASPECT_RATIOS) == 1 is true and aspect ratio list ASPECT_RATIOS[0] is used
    # for all IN_FEATURES.  # Default: [[0.5, 1.0, 2.0]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    # Anchor angles.
    # list[list[float]], the angle in degrees, for each input feature map.
    # ANGLES[i] specifies the list of angles for IN_FEATURES[i].
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[-90, 0, 90]]
    # Relative offset between the center of the first anchor and the top-left corner of the image
    # Value has to be in [0, 1). Recommend to use 0.5, which means half stride.
    # The value is not expected to affect model accuracy.
    cfg.MODEL.ANCHOR_GENERATOR.OFFSET = 0.0

    # ---------------------------------------------------------------------------- #
    # RPN options
    # ---------------------------------------------------------------------------- #
    #cfg.MODEL.RPN = CN()
    #cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHead"  # used by RPN_HEAD_REGISTRY

    # Names of the input feature maps to be used by RPN
    # e.g., ["p2", "p3", "p4", "p5", "p6"] for FPN
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    # Remove RPN anchors that go outside the image by BOUNDARY_THRESH pixels
    # Set to -1 or a large value, e.g. 100000, to disable pruning anchors
    cfg.MODEL.RPN.BOUNDARY_THRESH = -1
    # IOU overlap ratios [BG_IOU_THRESHOLD, FG_IOU_THRESHOLD]
    # Minimum overlap required between an anchor and ground-truth box for the
    # (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
    # ==> positive RPN example: 1)
    # Maximum overlap allowed between an anchor and ground-truth box for the
    # (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
    # ==> negative RPN example: 0)
    # Anchors with overlap in between (BG_IOU_THRESHOLD <= IoU < FG_IOU_THRESHOLD)
    # are ignored (-1)
    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
    cfg.MODEL.RPN.IOU_LABELS = [0, -1, 1]
    # Number of regions per image used to train RPN
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
    # Target fraction of foreground (positive) examples per RPN minibatch
    cfg.MODEL.RPN.POSITIVE_FRACTION = 0.5
    # Options are: "smooth_l1", "giou", "diou", "ciou"
    cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "smooth_l1"
    cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = 1.0
    # Weights on (dx, dy, dw, dh) for normalizing RPN anchor regression targets
    cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
    # The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
    cfg.MODEL.RPN.SMOOTH_L1_BETA = 0.0
    cfg.MODEL.RPN.LOSS_WEIGHT = 1.0
    # Number of top scoring RPN proposals to keep before applying NMS
    # When FPN is used, this is *per FPN level* (not total)
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
    # Number of top scoring RPN proposals to keep after applying NMS
    # When FPN is used, this limit is applied per level and then again to the union
    # of proposals from all levels
    # NOTE: When FPN is used, the meaning of this config is different from Detectron1.
    # It means per-batch topk in Detectron1, but per-image topk here.
    # See the "find_top_rpn_proposals" function for details.
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    # NMS threshold used on RPN proposals
    cfg.MODEL.RPN.NMS_THRESH = 0.7
    # Set this to -1 to use the same number of output channels as input channels.
    cfg.MODEL.RPN.CONV_DIMS = [-1]

    # ---------------------------------------------------------------------------- #
    # ROI HEADS options
    # ---------------------------------------------------------------------------- #
    #cfg.MODEL.ROI_HEADS = CN()
    #cfg.MODEL.ROI_HEADS.NAME = "Res5ROIHeads"

    # Number of foreground classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    # Names of the input feature maps to be used by ROI heads
    # Currently all heads (box, mask, ...) use the same input feature map list
    # e.g., ["p2", "p3", "p4", "p5"] is commonly used for FPN
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    # IOU overlap ratios [IOU_THRESHOLD]
    # Overlap threshold for an RoI to be considered background (if < IOU_THRESHOLD)
    # Overlap threshold for an RoI to be considered foreground (if >= IOU_THRESHOLD)
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
    cfg.MODEL.ROI_HEADS.IOU_LABELS = [0, 1]
    # RoI minibatch size *per image* (number of regions of interest [ROIs]) during training
    # Total number of RoIs per training minibatch =
    #   ROI_HEADS.BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH
    # E.g., a common configuration is: 512 * 16 = 8192
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    # Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

    # ---------------------------------------------------------------------------- #
    # Mask Head
    # ---------------------------------------------------------------------------- #
    #cfg.MODEL.ROI_MASK_HEAD = CN()
    #cfg.MODEL.ROI_MASK_HEAD.NAME = "MaskRCNNConvUpsampleHead"
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.ROI_MASK_HEAD.NUM_CONV = 0  # The number of convs in the mask head
    cfg.MODEL.ROI_MASK_HEAD.CONV_DIM = 256
    # Normalization method for the convolution layers.
    # Options: "" (no norm), "GN", "SyncBN".
    cfg.MODEL.ROI_MASK_HEAD.NORM = ""
    # Whether to use class agnostic for mask prediction
    cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = False
    # Type of pooling operation applied to the incoming feature map for each RoI
    cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE = "ROIAlignV2"

    f = open(cfg.OUTPUT_DIR+'/config.yml', 'w')
    f.write(cfg.dump())
    f.close()

    return cfg


def main(args):
    cfg = setup(args)

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
    for exp_idx in np.arange(START_IDX, len(experiment_titles)):
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

        launch(
            main,
            NUM_GPUS,
            num_machines=1,
            machine_rank=0,
            dist_url='tcp://127.0.0.1:5000'+str(np.random.randint(0, 9)),
            args=({"output_dir":output_dir, "dataset_dirs":exp_datasets, "num_gpus":NUM_GPUS, "gpu_batch_size":BATCH_SIZE, "augmentations":exp_augs, "IDX":exp_idx},),
        )

        gc.collect()
