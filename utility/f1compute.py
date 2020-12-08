from .common import get_iou
import torch
import numpy as np

def compute_f1(pred, gt):
    prec, recall = pres_recall(pred, gt)
    if recall == 0 or prec == 0:
        return 0
    return 2 * (recall * prec) / (recall + prec)

def pres_recall(pred, gt, thresh_iou=0.5):
    if len(pred) == 0 or len(gt) == 0:
        if len(pred) == len(gt): 
            return 1, 1
        else:
            return 0, 0
    table = np.zeros((len(pred), len(gt)))
    for i, b in enumerate(pred):
        for j, g in enumerate(gt):
            table[i, j] = get_iou(b, g)
    tp = 0.
    while True:
        index2d = np.unravel_index(table.argmax(), table.shape)
        if table[index2d] <= thresh_iou:
            break
        tp += 1
        table[index2d[0], :] = -1
        table[:, index2d[1]] = -1

    precision = tp / len(pred)
    recall = tp / len(gt)
    return (precision, recall)