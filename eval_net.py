import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch
import cv2

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from detectron2 import structures

from detectron2.evaluation import DatasetEvaluator

DISPLAY = False

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


class mAPEvaluator(DatasetEvaluator):
    def __init__(
            self,
            dataset_name,
            distributed=True,
            output_dir=None,
    ):
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self._poly_anns = DatasetCatalog.get(dataset_name)


    def reset(self):
        self._instance_cnt = []
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            img = input['image'].to(self._cpu_device).numpy().transpose([1, 2, 0])

            pred_scores = output["instances"].to(self._cpu_device).scores.numpy()
            pred_masks = output["instances"].to(self._cpu_device).pred_masks.numpy().copy()

            gt_bitmasks = np.zeros((1, pred_masks.shape[1], pred_masks.shape[2]), dtype=np.uint8)
            h_w_scalers = np.array([pred_masks.shape[2] / img.shape[1], pred_masks.shape[1] / img.shape[0]])
            if len(input['instances'].gt_boxes):
                gt_mask_polys = [poly[0].astype(np.float32).reshape(-1, 2) for poly in input['instances'].gt_masks]
                gt_bitmasks = np.zeros((len(gt_mask_polys), pred_masks.shape[1], pred_masks.shape[2]), dtype=np.uint8)
                for idx, poly in enumerate(gt_mask_polys):
                    cv2.fillPoly(gt_bitmasks[idx], np.int32([h_w_scalers*poly]), 1)
                self._instance_cnt.append(len(input['instances'].gt_boxes))

            gt_masks = gt_bitmasks.astype(bool)
            masks_pred_rle = mask_util.encode(np.asfortranarray(pred_masks.transpose([1, 2, 0])))
            masks_gt_rle = mask_util.encode(np.asfortranarray(gt_bitmasks.transpose([1, 2, 0])))
            MASK_IOUs = mask_util.iou(masks_gt_rle, masks_pred_rle, gt_masks.shape[0]*[0])
            #print(MASK_IOUs)

            result_tuples = list(zip(pred_scores, TPs(MASK_IOUs)[:]))
            self._predictions.extend(result_tuples)

            if DISPLAY:
                cv2.imshow("Input", img)
                gt_img = gt_bitmasks.sum(axis=0)[:, :, None].astype(float)
                pred_img = pred_masks.astype(np.float32).sum(axis=0)[:, :, None]
                cv2.imshow("Output + GT", cv2.resize((gt_img + pred_img)/2, (2000, 1000)))
                cv2.waitKey()


    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):
        * Mean intersection-over-union averaged across classes (mIoU)
        """
        if self._distributed:
            synchronize()
            self._instance_cnt = sum(all_gather(self._instance_cnt), [])
            self._predictions = sum(all_gather(self._predictions), [])
            if not is_main_process():
                return

        self._predictions.sort(reverse=True, key=lambda a: a[0])
        instance_cnt = 0
        for inst_cnts in self._instance_cnt:
            instance_cnt += inst_cnts
        p_sum = 0
        mask_tp_sum = 0
        mask_p_l = []
        mask_r_l = []
        for prediction in self._predictions:
            p_sum += 1
            mask_tp_sum += prediction[1].astype(int)
            mask_p_l.append(mask_tp_sum / p_sum)
            mask_r_l.append(mask_tp_sum / instance_cnt)

        res = {}
        maps = upperAUC(np.array(mask_r_l), np.array(mask_p_l))
        res[self._dataset_name + " mask mAP[0.5:0.95]"] = np.mean(maps)
        print(self._dataset_name + " mask mAP[0.5:0.95]: " + str(maps))

        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results
