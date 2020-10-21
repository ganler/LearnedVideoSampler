# Copyright (c) 2020 Ganler
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import cv2
from typing import List
import torch
import numpy as np

def opticalflow(l, r):
    l = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
    r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(l, r, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def concat3channel2tensor(l, r, batch_dim=False):
    l = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
    r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    im = np.zeros((*l.shape[:2], 3))
    im[:, :, 0] = l
    im[:, :, 1] = r
    return totensor(im, wh=l.shape[1::-1], batch_dim=batch_dim)

def opticalflow2tensor(l, r, batch_dim=False):
    flow = opticalflow(l, r)
    im = np.zeros((*l.shape[:2], 3))
    im[:, :, :2] += flow
    return totensor(im, wh=l.shape[1::-1], batch_dim=batch_dim)

@torch.no_grad()
def totensor(im, wh, batch_dim=False):
    # No need to convert BGR to RGB for optical flow
    im = cv2.resize(im, wh)
    inp = im.transpose(2, 0, 1) # W H C => C W H
    inp = np.ascontiguousarray(inp, dtype=np.float32)
    inp = torch.from_numpy(inp) / 255.

    if batch_dim and inp.ndimension() == 3:
        inp = inp.unsqueeze(0)

    return inp

@torch.no_grad()
def _boxlist2tensor(boxlist: torch.Tensor, tensor_resolution=(608, 352), factor=2, use_conf=False) -> torch.Tensor:
    # Input : List[[BoxIndex, X, Y, W, H, CONF]]
    # Output: [SequenceIndex, 1, TensorHeight, TensorWidth]
    ret = np.zeros((tensor_resolution[1] // factor, tensor_resolution[0] // factor))
    if 0 != boxlist.nelement():
        boxlist = boxlist[:, :5].cpu().numpy()
        for (*xyxy, conf) in boxlist:
            intxyxy = [int(element / factor) for element in xyxy]
            (x0, y0, x1, y1) = intxyxy
            ret[y0:(y1+1), x0:(x1+1)] += conf if use_conf else 1.
    return torch.from_numpy(ret)

@torch.no_grad()
def _boxlist2tensor_channelstack(boxlists: torch.Tensor, tensor_resolution=(608, 352), factor=4, use_conf=False) -> torch.Tensor:
    # Input : List[[BoxIndex, X, Y, W, H, CONF]]
    # Output: [SequenceIndex, 1, TensorHeight, TensorWidth]
    ret = np.zeros((len(boxlists), tensor_resolution[1] // factor, tensor_resolution[0] // factor), dtype=np.float32)
    for index, boxlist in enumerate(boxlists):
        if 0 != boxlist.nelement():
            boxlist = boxlist[:, :5].cpu().numpy()
            for (*xyxy, conf) in boxlist:
                intxyxy = [int(element / factor) for element in xyxy]
                (x0, y0, x1, y1) = intxyxy
                ret[index][y0:(y1+1), x0:(x1+1)] += conf if use_conf else 1.
    return torch.from_numpy(ret)

@torch.no_grad()
def _boxembedding(boxlists, top_n=16) -> torch.Tensor:
    # Input : List[[BoxIndex, X, Y, W, H, CONF]]
    # Output: [SequenceIndex, 1, TensorHeight, TensorWidth]
    ret = torch.zeros((len(boxlists), top_n, 5))
    for index, boxlist in enumerate(boxlists):
        if 0 != boxlist.nelement():
            for i in range(min(top_n, len(boxlist))):
                ret[index][i] = boxlist[i, :5]
    return ret

@torch.no_grad()
def boxembeddingpair(l, r, top_n=16):
    ret = _boxembedding([l, r], top_n=top_n).unsqueeze(0)
    return ret

@torch.no_grad()
def boxlist2tensor(l, r, tensor_res=(608, 352), batch_dim=False):
    # L & R are box lists.
    l = _boxlist2tensor(l, tensor_res)
    r = _boxlist2tensor(r, tensor_res)
    ret = torch.zeros(3, l.shape[0], l.shape[1])

    ret[0] += l
    ret[1] += r

    if batch_dim and ret.ndimension() == 3:
        ret = ret.unsqueeze(0)
    
    return ret

