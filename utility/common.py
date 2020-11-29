import argparse
from typing import no_type_check
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def expected_distribution(label, n_opt):
    '''
    label: [B]
    '''
    ret = torch.arange(n_opt, dtype=torch.float).unsqueeze(0).repeat(len(label), 1).transpose(0, -1)
    ret = ret.to(label.device)
    ret = - (ret - label).transpose(0, -1).abs() + n_opt
    return ret / ret.sum(axis=1).reshape(-1, 1)

class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

class distance_loss:
    def __init__(self, n_opt):
        self.n_opt = n_opt
    
    def __call__(self, pred, label):
        '''
        pred: [B, N_OPT]
        label: [B]
        '''
        tar = expected_distribution(label, self.n_opt)
        ret = F.mse_loss(pred, tar)
        # print(pred)
        # print(label)
        # print(ret)
        return ret


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

def iou_pairing(l: torch.Tensor, r: torch.Tensor, negconst=-5, inf=1000, use_index=False) -> np.array:
    '''
    || Output: pairs...
    '''
    ret = []
    paired_l = []
    paired_r = []

    buyers, gifts = None, None
    if len(l) != 0:
        buyers = l.numpy()[:, :4]  # xyxy

    if len(r) != 0:
        gifts = r.numpy()[:, :4]

    if len(l) != 0 and len(r) != 0:
        table = np.zeros((len(buyers), len(gifts)))

        for i, b in enumerate(buyers):
            for j, g in enumerate(gifts):
                table[i, j] = get_iou(b, g)

        while True:
            index2d = np.unravel_index(table.argmax(), table.shape)
            if table[index2d] <= 0:
                break
            from_ = buyers[index2d[0]]
            to_ = gifts[index2d[1]]
            # [x, y, dx, dy, iou]
            ret.append([
                (from_[0] + from_[2]) / 2,
                (from_[1] + from_[3]) / 2,
                (from_[1] + from_[3] - to_[0] - to_[2]) / 2,
                (from_[1] + from_[3] - to_[1] - to_[3]) / 2,
                table[index2d] if not use_index else index2d
            ])
            paired_l.append(index2d[0])
            paired_r.append(index2d[1])
            table[index2d[0], :] = -1
            table[:, index2d[1]] = -1

    for ll in range(len(l)):
        if ll not in paired_l:
            from_ = buyers[ll]
            ret.append([
                (from_[0] + from_[2]) / 2,
                (from_[1] + from_[3]) / 2,
                inf,
                inf,
                negconst if not use_index else (ll, -1)
            ])

    for rr in range(len(r)):
        if rr not in paired_r:
            to_ = gifts[rr]
            ret.append([
                (to_[0] + to_[2]) / 2,
                (to_[1] + to_[3]) / 2,
                -inf,
                -inf,
                negconst if not use_index else (-1, rr)
            ])
    return np.array(ret)

class iou_pairing_skipper:
    def __init__(self, conf_thresh=0.5):
        self.conf_thresh = conf_thresh

    def judge(self, l, r):
        out = np.array(iou_pairing(l, r))
        if len(out) == 0:
            return True
        return np.average(out[:, -1]) > self.conf_thresh
