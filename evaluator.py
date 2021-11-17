from detectron2.engine import DefaultPredictor
import pycocotools.mask as mask_utils
from detectron2 import structures
import torch
import cv2
import os
import Params
import numpy as np
import json

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

def test_experiment(cfg, test_dirs, output_dir):
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0
    predictor = DefaultPredictor(cfg)
    
    for test_dir in test_dirs:
        dir_name = test_dir.split('/')[-1]
        with open(test_dir + "/labels.json", 'r') as fp:
            dataset_dict = json.load(fp)
    
        prediction_results = []
        instance_cnt = 0
        print("Calculating AP@[0.5:0.95]...")
        for data_inst in dataset_dict:
            gt_anns = data_inst['annotations']
            gt_bboxes = np.array([inst['bbox'] for inst in gt_anns if inst['name'] == 'scallop'])
            im = cv2.imread(data_inst['file_name'])
            outputs = predictor(im)
            pred_scores = outputs["instances"].to("cpu").scores.numpy()
            bboxes_gt = structures.Boxes(torch.Tensor(gt_bboxes))
            bboxes_pred = outputs["instances"].to("cpu").pred_boxes
            BOX_IOUs = structures.pairwise_iou(bboxes_gt, bboxes_pred).cpu().detach().numpy()
    
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
    
            if len(BOX_IOUs) == 0 or len(MASK_IOUs) == 0:
                continue
            assert (BOX_IOUs.shape == MASK_IOUs.shape)
            if BOX_IOUs.shape[0] and pred_scores.shape[0]:
                result_tuples = list(zip(pred_scores, TPs(BOX_IOUs)[:], TPs(MASK_IOUs)[:]))
                prediction_results.extend(result_tuples)
            instance_cnt += BOX_IOUs.shape[0]
    
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
    
        results_dict = {'mAPs[0.5:0.95]: [BBox, Mask]':[upperAUC(np.array(bbox_R_l), np.array(bbox_P_l)).tolist(), upperAUC(np.array(mask_R_l), np.array(mask_P_l)).tolist()]}
                        # 'Scallop BBox [R, P]':[bbox_R_l, bbox_P_l],
                        # 'Scallop Mask [R, P]':[mask_R_l, mask_P_l]}
        print(results_dict)
        with open(output_dir + '/' + dir_name + '_eval_results.json', 'w') as output_file:
            output_file.write(json.dumps(results_dict))

    del predictor
