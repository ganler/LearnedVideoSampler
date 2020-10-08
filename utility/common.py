import argparse
import numpy as np
import torch

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou

def iou_pairing(l: torch.Tensor, r: torch.Tensor) -> np.array:
    assert len(l) == len(r)

    if len(l) == 0:
        return np.array([])

    l = l.numpy()[:, :4]
    r = r.numpy()[:, :4]
    occ_r = []
    ret = np.zeros((len(l), 3))
    for i, ll in enumerate(l):
        best_j = 0
        best_iou = -10000
        for j, rr in enumerate(r):
            if j in occ_r:
                continue
            cur_iou = get_iou(ll, rr)
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_j = j
        occ_r.append(best_j)
        ret[i] = [i, best_j, best_iou]
    return ret

class iou_pairing_skipper:
    def __init__(self, conf_thresh=0.5):
        self.conf_thresh = conf_thresh

    def judge(self, l, r):
        out = iou_pairing(l, r)
        if len(out) == 0:
            return True
        return np.average(out[:, 2]) > self.conf_thresh
