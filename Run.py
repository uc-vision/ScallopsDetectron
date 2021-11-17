from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog, transforms
import pycocotools.mask as mask_utils
from detectron2 import structures
import torch
import cv2
import os
import pathlib
import Params as P
import numpy as np
import json
import re
from tqdm import tqdm
from itertools import groupby
import matplotlib.pyplot as plt

SHOW = True
WAIT_DELAY = 1

DS_DIV = 1
TEST_IMG_DIR = "/local/ScallopMaskDataset/train_lr/"# "/home/cosc/research/CVlab/bluerov_data/210113-065012/"#"/local/ScallopMaskDataset/train/"#
MODEL_PATH = "/local/ScallopMaskRCNNOutputs/HR label prop data AUGS/"

paths = pathlib.Path(TEST_IMG_DIR)
if any(fname.endswith('.pgm') for fname in os.listdir(TEST_IMG_DIR)):
    img_fns = [str(fn) for fn in paths.iterdir() if str(fn)[-4:] == '.pgm']
    BAYER = True
else:
    img_fns = [str(fn) for fn in paths.iterdir() if str(fn)[-4:] == '.jpg' or str(fn)[-4:] == '.png']
    BAYER = False
print("Num images in dir: {}".format(len(img_fns)))
img_fns.sort(key=lambda f: int(re.sub('\D', '', f)))


def readImg(path):
    im = cv2.imread(path)
    if BAYER:
        im = cv2.cvtColor(im[:, :, 0][:, :, None], cv2.COLOR_BAYER_BG2BGR)
    return im

def showOutput(img, output, gt=None, waitkey=0):
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
    v = v.draw_instance_predictions(output["instances"].to("cpu"))
    out_image = v.get_image()[:, :, ::-1]
    cv2.imshow("Detectron input", cv2.resize(im, (im.shape[1]//DS_DIV, im.shape[0]//DS_DIV)))
    cv2.imshow("Detectron output", cv2.resize(out_image, (out_image.shape[1]//DS_DIV, out_image.shape[0]//DS_DIV)))

    if gt is not None:
        visualizer = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
        vis = visualizer.draw_dataset_dict(gt)
        gt_img = vis.get_image()[:, :, ::-1]
        cv2.imshow("Ground truth", cv2.resize(gt_img, (gt_img.shape[1]//DS_DIV, gt_img.shape[0]//DS_DIV)))
        cv2.imshow("Overlap", cv2.resize(gt_img, (gt_img.shape[1]//DS_DIV, gt_img.shape[0]//DS_DIV))/500 + cv2.resize(out_image, (out_image.shape[1]//DS_DIV, out_image.shape[0]//DS_DIV))/500)
    return cv2.waitKey(waitkey) == ord('q')

AP_METS = np.arange(0.5, 1, 0.05)
def TPs(ious):
    iou_maxes = np.max(ious, axis=0)
    iou_arr = iou_maxes[None].repeat(len(AP_METS), axis=0)
    ap_thresh_mask = iou_arr > AP_METS[:, None]
    return ap_thresh_mask.transpose()

def upperAUC(x_arr, y_arr):
    dx = x_arr[1:] - x_arr[:-1]
    y = y_arr[1:] * (y_arr[1:] >= y_arr[:-1])
    return (dx * y).sum(axis=0)

cfg = get_cfg()
cfg.NUM_GPUS = 1
cfg.merge_from_file(MODEL_PATH + 'config.yml')
model_paths = [str(path) for path in pathlib.Path(MODEL_PATH).glob('*.pth')]
model_paths.sort()
cfg.MODEL.WEIGHTS = os.path.join(model_paths[-1])
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0
predictor = DefaultPredictor(cfg)

dataset_dict = None
if any(fname.endswith('.json') for fname in os.listdir(TEST_IMG_DIR)):
    with open(TEST_IMG_DIR + "/labels.json", 'r') as fp:
        dataset_dict = json.load(fp)
        for data_entry in dataset_dict:
            data_entry["file_name"] = TEST_IMG_DIR + '/' + data_entry["file_name"].split('/')[-1]


bbox_TPs = np.zeros((len(AP_METS)))
mask_TPs = np.zeros((len(AP_METS)))
total_positives = np.zeros((len(AP_METS)))
total_instances = np.zeros((len(AP_METS)))

prediction_results = []
instance_cnt = 0
if dataset_dict is not None:
    print("Calculating AP@[0.5:0.95]...")
    for data_inst in tqdm(dataset_dict[::10]):
        gt_anns = data_inst['annotations']
        gt_bboxes = np.array([inst['bbox'] for inst in gt_anns if inst['name'] == 'scallop'])
        im = readImg(data_inst['file_name'])
        outputs = predictor(im)
        pred_scores = outputs["instances"].to("cpu").scores.numpy()
        bboxes_gt = structures.Boxes(torch.Tensor(gt_bboxes))
        bboxes_pred = outputs["instances"].to("cpu").pred_boxes
        BOX_IOUs = structures.pairwise_iou(bboxes_gt, bboxes_pred).cpu().detach().numpy()
        #print("BBOX IOUs: {}".format(BOX_IOUs))

        gt_mask_polys = [np.array(inst['segmentation'][0]).astype(np.float32) for inst in gt_anns if inst['name'] == 'scallop']
        gt_bitmasks = np.zeros((len(gt_mask_polys), im.shape[0], im.shape[1]))
        for idx, poly in enumerate(gt_mask_polys):
            cv2.fillPoly(gt_bitmasks[idx], np.int32([poly]), 1)
        gt_bitmasks = gt_bitmasks.astype(bool)
        pred_bitmasks = outputs["instances"].to("cpu").pred_masks.numpy()
        masks_pred_rle = mask_utils.encode(np.asfortranarray(pred_bitmasks.transpose([1, 2, 0])))
        masks_gt_rle = mask_utils.encode(np.asfortranarray(gt_bitmasks.transpose([1, 2, 0])))
        iscrowd = [int(inst['iscrowd']) for inst in gt_anns if inst['name'] == 'scallop']
        MASK_IOUs = mask_utils.iou(masks_gt_rle, masks_pred_rle, iscrowd)
        #print("MASK IOUs: {}".format(MASK_IOUs))

        if not(len(BOX_IOUs) == 0 or len(MASK_IOUs) == 0):
            assert (BOX_IOUs.shape == MASK_IOUs.shape)
            if BOX_IOUs.shape[0] and pred_scores.shape[0]:
                result_tuples = list(zip(pred_scores, TPs(BOX_IOUs)[:], TPs(MASK_IOUs)[:]))
                prediction_results.extend(result_tuples)
            instance_cnt += BOX_IOUs.shape[0]

        if SHOW:
            if showOutput(im, outputs, gt=data_inst, waitkey=WAIT_DELAY):
                break

    prediction_results.sort(reverse=True, key=lambda a: a[0])
    P_sum = 0
    mask_TP_sum = 0
    mask_P_l = []
    mask_R_l = []
    bbox_TP_sum = 0
    bbox_P_l = []
    bbox_R_l = []
    for prediction in prediction_results:
        P_sum += 1
        bbox_TP_sum += prediction[1].astype(int)
        bbox_P_l.append(bbox_TP_sum / P_sum)
        bbox_R_l.append(bbox_TP_sum / instance_cnt)
        mask_TP_sum += prediction[2].astype(int)
        mask_P_l.append(mask_TP_sum / P_sum)
        mask_R_l.append(mask_TP_sum / instance_cnt)

    title_keys = ['Scallop BBox', 'Scallop Mask']
    arrays = [[np.array(bbox_R_l), np.array(bbox_P_l)], [np.array(mask_R_l), np.array(mask_P_l)]]
    mAPs = []
    for plot_i in range(2):
        plt.figure()
        x_arr, y_arr = arrays[plot_i]
        mAPs.append(upperAUC(x_arr, y_arr))
        print("{} mAPs: {}".format(title_keys[plot_i], mAPs[plot_i]))
        print("{} mAP_coco: {}".format(title_keys[plot_i], mAPs[plot_i].sum()/AP_METS.shape[0]))
        plt.title(title_keys[plot_i] + " Precision-recall curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        legend = []
        plt.grid(True)
        for idx, ap_met in enumerate(AP_METS):
            plt.plot(x_arr[:, idx], y_arr[:, idx])
            legend.append('AP'+str(round(ap_met*100)))
        plt.legend(legend, loc='upper right')
    plt.show()

else:
    for path in img_fns:
        im = readImg(path)
        outputs = predictor(im)
        if showOutput(im, outputs):
            break
