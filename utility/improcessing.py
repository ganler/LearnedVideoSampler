# Copyright (c) 2020 Ganler
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import cv2
import torch
import numpy as np

def opticalflow(l, r):
    l = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
    r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(l, r, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

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