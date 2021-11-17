from detectron2.data import DatasetCatalog
from detectron2.data import DatasetMapper
import numpy as np
import torch
import os
import cv2
from termcolor import colored
from detectron2.utils.logger import logging, log_every_n_seconds
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import Visualizer
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from detectron2.engine import launch
logger = logging.getLogger("detectron2")

VALID_GRAD_SAMPLES = 10
VALID_GRAD_CUTOFF = -0.2 # Per Eval period (10000)

DISPLAY = False
DELAY = 0
if DISPLAY:
    cv2.namedWindow("Training Instance", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Test Instance", cv2.WINDOW_NORMAL)


def do_test(cfg, model, storage):
    print("Evaluating...")
    dataset_loss_mean = 0
    num_datasets = 0
    for dataset_name in cfg.DATASETS.TEST:
        losses = []
        num_datasets += 1
        mapper = DatasetMapper(cfg, is_train=True)
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        total = len(data_loader)
        num_warmup = min(5, total - 1)

        for idx, inputs in enumerate(data_loader):
            print("FN: {}, TEST IDX: {}".format(inputs[0]["file_name"], idx), end='\r')
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if idx >= num_warmup * 2:
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}.".format(
                        idx + 1, total), n=5)
            metrics_dict = model(inputs)
            metrics_dict = {
                k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
                for k, v in metrics_dict.items()
            }
            loss_batch = sum(loss for loss in metrics_dict.values())
            losses.append(loss_batch)

            if DISPLAY:
                for d in inputs:
                    img = d["image"].cpu().numpy().transpose([1, 2, 0])
                    v = Visualizer(img[:, :, ::-1])
                    if d["instances"].__len__() > 0:
                        v = v.overlay_instances(masks=d["instances"].gt_masks, boxes=d["instances"].gt_boxes)
                        img = v.get_image()[:, :, ::-1]
                    cv2.imshow("Test Instance", img)
                    if cv2.waitKey(DELAY) == ord('q'):
                        exit(0)

        mean_loss = np.mean(losses)
        dataset_loss_mean += mean_loss
        if not DISPLAY:
            storage.put_scalar(dataset_name+'_loss', mean_loss)
            comm.synchronize()
        print("\nValidation Loss: {}".format(mean_loss))
    return dataset_loss_mean / max(1, num_datasets)


def do_train(cfg, model, augs, resume=False):
    # if __name__ == '__main__':
    #     torch.multiprocessing.freeze_support()
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
            checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    mapper = DatasetMapper(cfg, is_train=True, augmentations=augs)
    data_loader = build_detection_train_loader(cfg, mapper=mapper)
    logger.info("Starting training from iteration {}".format(start_iter))
    valid_metric_arr = np.array([])
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            print("Iteration: {}, FN: {}".format(iteration, data[0]["file_name"]))
            if DISPLAY:
                print("Iteration: {}".format(iteration))
                for d in data:
                    print(d["file_name"])
                    img = d["image"].cpu().numpy().transpose([1, 2, 0])
                    v = Visualizer(img[:, :, ::-1]) #, MetadataCatalog.get(cfg.DATASETS.TRAIN), scale=1
                    #print(d["instances"])
                    if d["instances"].__len__() > 0:
                        v = v.overlay_instances(masks=d["instances"].gt_masks, boxes=d["instances"].gt_boxes)
                        img = v.get_image()[:, :, ::-1]
                    cv2.imshow("Training Instance", img)
                    if cv2.waitKey(DELAY) == ord('q'):
                        exit(0)

            iteration = iteration + 1
            storage.step()

            loss_dict = model(data)

            losses = sum(loss for loss in loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            if iteration % 50 == 0:
                print(colored("Itteration {}; Total loss: {}, Loss dict: {}".format(iteration, losses_reduced, loss_dict_reduced), 'green'))
            if not DISPLAY and comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            if not DISPLAY:
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and iteration % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter
            ):
                valid_metric_arr = np.append(valid_metric_arr, do_test(cfg, model, storage))
                if not DISPLAY:
                    comm.synchronize()

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

            # Early stop condition based on validation metric gradient
            if len(valid_metric_arr) > VALID_GRAD_SAMPLES:
                met_grads = valid_metric_arr[1:] - valid_metric_arr[:-1]
                valid_metric_arr = valid_metric_arr[1:].copy()
                avg_grad = np.mean(met_grads)
                print("AVG validation metric gradient: {}".format(avg_grad))
                if avg_grad > VALID_GRAD_CUTOFF:
                    print("Early stop condition met at {} itterations.".format(iteration))
                    break


def train_experiment(cfg, datasets, augmentations, output_dir):
    train_dirs, valid_dirs = datasets

    # Changes from default:
    # Changed dataset pixel mean
    # Disabled input crop and flip

    cfg.OUTPUT_DIR = output_dir
    cfg.DATASETS.TRAIN = train_dirs
    cfg.DATASETS.TEST = valid_dirs
    cfg.TEST.EVAL_PERIOD = 1000
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.CHECKPOINT_PERIOD = 10000

    cfg.NUM_GPUS = 4
    cfg.SOLVER.IMS_PER_BATCH = 4 * cfg.NUM_GPUS
    cfg.SOLVER.REFERENCE_WORLD_SIZE = 4

    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.GAMMA = 0.1
    # The iteration number to decrease learning rate by GAMMA.
    cfg.SOLVER.STEPS = (10000,)

    cfg.SOLVER.MAX_ITER = 100000
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    # The weight decay that's applied to parameters of normalization layers
    # (typically the affine transformation)
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0

    cfg.MODEL.BACKBONE.FREEZE_AT = 0

    # Default from ImageNet dataset: [103.530, 116.280, 123.675]
    cfg.MODEL.PIXEL_MEAN = [125.04907649, 125.30469809,  85.26628485]
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

    print(cfg)
    if not DISPLAY:
        f = open(output_dir+'/config.yml', 'w')
        f.write(cfg.dump())
        f.close()

    #model = build_model(cfg)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    #trainer.train()
    #ddp_model = DDP(model, device_ids=[0])

    launch(trainer.train, num_gpus_per_machine=4, dist_url="auto")

    #do_train(cfg, model, augmentations, resume=False)

    # if not DISPLAY:
    #     f = open(output_dir+'/config.yml', 'w')
    #     f.write(cfg.dump())
    #     f.close()

    model.cpu()
    del model