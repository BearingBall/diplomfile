from cv2 import threshold
import torch
import numpy as np


def general_objectness(labels, device):  # as in InvisibleCloak
    score = torch.tensor(0.0).to(device)
    # score.requires_grad = True
    for i in range(len(labels["labels"])):
        if labels["labels"][i] == 1: # and labels["scores"][i] > 0.6
            score += max(labels["scores"][i] + 1, 0)**2
    return score


def total_variation(patch):  # TV - total variation penalthy (smooth for patch)
    a1 = patch[:, :, :-1] - patch[:, :, 1:]
    a2 = patch[:, :-1, :] - patch[:, 1:, :]
    return a1.abs().sum() + a2.abs().sum()


def intersection_over_union(boxes_preds, boxes_labels):
    box1_x1 = boxes_preds[..., 0:1] 
    box1_y1 = boxes_preds[..., 1:2]
    box1_x2 = boxes_preds[..., 2:3]
    box1_y2 = boxes_preds[..., 3:4]

    box2_x1 = boxes_labels[..., 0:1]
    box2_y1 = boxes_labels[..., 1:2]
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3]
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def TP_FP(predictions, true_boxes, score_threshold, iou_threshold):
    TP = 0
    for i in range(len(predictions['boxes'])):
        for j in range(len(true_boxes)):
            if predictions['scores'][i] > score_threshold and intersection_over_union(predictions['boxes'][i], true_boxes[j]) > iou_threshold:
                TP += 1
                break
    return TP, len(predictions['boxes']) - TP


def FN(predictions, true_boxes, score_threshold, iou_threshold):
    FN = 0
    for j in range(len(true_boxes)):
        FN += 1
        for i in range(len(predictions['boxes'])):
            if predictions['scores'][i] > score_threshold and intersection_over_union(predictions['boxes'][i], true_boxes[j]) > iou_threshold:
                FN -=1
                break
    return FN


def precision(predictions, true_boxes, score_threshold, iou_threshold):
    tp, fp = TP_FP(predictions, true_boxes, score_threshold, iou_threshold)
    return tp / (tp + fp + 1e-6)


def mean_average_precision(predictions, true_boxes, iou_threshold=0.5):
    thresholds = np.arange(0.0, 1.0, 0.1)
    precisions = []
    for tr in thresholds:
        precisions.append(precision(predictions, true_boxes, tr, iou_threshold))
    return np.mean(np.asarray(precisions))

