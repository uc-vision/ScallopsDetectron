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
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt


AP_METS = np.arange(0.5, 1, 0.05)
def TPs(ious):
    iou_maxes = np.max(ious, axis=0)
    iou_arr = iou_maxes[None].repeat(len(AP_METS), axis=0)
    ap_thresh_mask = iou_arr >= AP_METS[:, None]
    return ap_thresh_mask.transpose()


def AUC(r_arr, p_arr):
    AP_sum = np.zeros_like(AP_METS)
    print(np.max(p_arr[1:], axis=0))
    for idx in range(r_arr.shape[0]-1):
        AP_sum += (r_arr[idx+1] - r_arr[idx]) * np.max(p_arr[idx+1:], axis=0)
    return AP_sum


def PIAP(r_arr, p_arr, n=11):
    AP_results = []
    for iou_idx in range(len(AP_METS)):
        r_a = r_arr[:, iou_idx]
        p_a = p_arr[:, iou_idx]
        p_sum = 0
        for r_val in np.linspace(0, 1, n):
            p_above_r = p_a[r_a >= r_val]
            if len(p_above_r):
                p_sum += np.max(p_above_r)
        AP_results.append(p_sum / n)
    return AP_results


# Beta is relative importance of recall to precision
def FScore(P_arr, R_arr, beta=1.0):
    return (1 + beta**2) * (P_arr * R_arr) / (beta**2 * P_arr + R_arr)


class mAPEvaluator(DatasetEvaluator):
    def __init__(
            self,
            dataset_name,
            display=False,
            distributed=True,
            output_dir=None,
    ):
        self._display = display
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
            pred_masks = output["instances"].to(self._cpu_device).pred_masks.numpy()
            pred_bboxes = output["instances"].to(self._cpu_device).pred_boxes

            gt_bboxes = input['instances'].gt_boxes
            if len(gt_bboxes) and len(pred_bboxes):
                self._instance_cnt.append(len(input['instances'].gt_boxes))
                gt_masks = structures.BitMasks.from_polygon_masks(input['instances'].gt_masks, input["height"], input["width"])
                masks_pred_rle = mask_util.encode(np.asfortranarray(pred_masks.transpose([1, 2, 0])))
                masks_gt_rle = mask_util.encode(np.asfortranarray(gt_masks.tensor.numpy().transpose([1, 2, 0])))
                MASK_IOUs = mask_util.iou(masks_gt_rle, masks_pred_rle, len(gt_bboxes)*[0])
                BOX_IOUs = structures.pairwise_iou(gt_bboxes, pred_bboxes).cpu().detach().numpy()
                result_tuples = list(zip(pred_scores, TPs(BOX_IOUs), TPs(MASK_IOUs)))
            else:
                F_arr = len(pred_scores)*[len(AP_METS)*[False]]
                result_tuples = list(zip(pred_scores, F_arr, F_arr))
            self._predictions.extend(result_tuples)

            if self._display:
                cv2.namedWindow("GT", cv2.WINDOW_GUI_NORMAL)
                cv2.namedWindow("GT+Pred", cv2.WINDOW_GUI_NORMAL)
                cv2.namedWindow("Input", cv2.WINDOW_GUI_NORMAL)
                v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self._dataset_name), scale=1)
                target_fields = input["instances"].get_fields()
                labels = [MetadataCatalog.get(self._dataset_name).thing_classes[i] for i in target_fields["gt_classes"]]
                vis = v.overlay_instances(
                    labels=labels,
                    boxes=target_fields.get("gt_boxes", None),
                    masks=target_fields.get("gt_masks", None),
                    keypoints=target_fields.get("gt_keypoints", None),
                )
                gt_image = vis.get_image()[:, :, ::-1]

                gtpred_img = gt_image.copy()
                v = Visualizer(gtpred_img[:, :, ::-1], MetadataCatalog.get(self._dataset_name), scale=1)
                vis = v.draw_instance_predictions(output["instances"].to(self._cpu_device))
                gtpred_img = vis.get_image()[:, :, ::-1]
                cv2.imshow("GT", gt_image)
                cv2.imshow("GT+Pred", gtpred_img)
                cv2.imshow("Input", img)
                #gt_img = gt_bitmasks.sum(axis=0)[:, :, None].astype(float)
                #pred_img = pred_masks.astype(np.float32).sum(axis=0)[:, :, None]
                #cv2.imshow("Output + GT", cv2.resize((gt_img + pred_img)/2, (2000, 1000)))
                cv2.waitKey()


    def evaluate(self, display=False):
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
        mask_tp_sum = np.zeros((len(AP_METS)))
        box_tp_sum = np.zeros((len(AP_METS)))
        mask_p_l = []
        mask_r_l = []
        box_p_l = []
        box_r_l = []
        CTP_list = []
        for prediction in self._predictions:
            CTP_list.append([prediction[0], prediction[1][0], prediction[2][0]])
            p_sum += 1
            box_tp_sum += np.array(prediction[1]).astype(int)
            mask_tp_sum += np.array(prediction[2]).astype(int)
            mask_p_l.append(mask_tp_sum / p_sum)
            mask_r_l.append(mask_tp_sum / instance_cnt)
            box_p_l.append(box_tp_sum / p_sum)
            box_r_l.append(box_tp_sum / instance_cnt)

        CTP_arr = np.array(CTP_list).astype(np.float32)
        precisions = np.cumsum(CTP_arr[:, 1:], axis=0) / np.arange(1, CTP_arr.shape[0]+1)[:, None]
        recalls = np.cumsum(CTP_arr[:, 1:], axis=0) / instance_cnt
        F1_scores = FScore(precisions, recalls, beta=1)
        max_F1_box_idx = np.argmax(F1_scores[:, 0])
        max_F1_mask_idx = np.argmax(F1_scores[:, 1])

        mask_maps = PIAP(np.array(mask_r_l), np.array(mask_p_l), n=101)
        bbox_maps = PIAP(np.array(box_r_l), np.array(box_p_l), n=101)
        bbox_APVOC = PIAP(np.array(box_r_l), np.array(box_p_l), n=11)
        
        print("BBox PASCAL VOC AP for IoU [0.5:0.95]: {}".format(bbox_APVOC))

        print("Max Bbox F1: {}, mAP: {}, P: {}, R: {}".format(F1_scores[max_F1_box_idx, 0], np.mean(bbox_maps),
                                                              precisions[max_F1_box_idx, 0], recalls[max_F1_box_idx, 0]))
        print(self._dataset_name + " bbox mAP[0.5:0.95]: " + str(bbox_maps))
        print("Confidence value for max mask F1: {}".format(CTP_arr[max_F1_mask_idx, 0]))
        print("Max Mask F1: {}, mAP: {}, P: {}, R: {}".format(F1_scores[max_F1_mask_idx, 1], np.mean(mask_maps),
                                                              precisions[max_F1_box_idx, 1], recalls[max_F1_box_idx, 1]))
        print(self._dataset_name + " mask mAP[0.5:0.95]: " + str(mask_maps))

        if display:
            titles = ["Mask ", "Bbox "]
            arrays = [[mask_r_l, mask_p_l], [box_r_l, box_p_l]]
            for a_idx, title in enumerate(titles):
                plt.figure()
                plt.title(title+"Precision-Recall curve")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                legend = []
                r_l, p_l = arrays[a_idx]
                plt.grid(True)
                for idx, ap_met in enumerate(AP_METS):
                    x_arr = list(np.array(r_l)[:, idx])
                    y_arr = list(np.array(p_l)[:, idx])
                    x_arr.append(x_arr[-1])
                    y_arr.append(0)
                    plt.plot(x_arr, y_arr)
                    legend.append('AP'+str(round(ap_met*100)))
                plt.legend(legend, loc='upper right')
            plt.show()

        res = {}
        res[self._dataset_name + " bbox mAP[0.5:0.95]"] = np.mean(bbox_maps)
        res[self._dataset_name + " mask mAP[0.5:0.95]"] = np.mean(mask_maps)
        #res[self._dataset_name + " Max mask F1 score"] = max_F1_mask_idx

        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results
