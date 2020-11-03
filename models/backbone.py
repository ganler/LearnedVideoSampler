from abc import ABC

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18
from typing import List
import torch.nn.functional as F

'''
Input signal can be
=> current image.
=> historical input data. 
'''


@torch.no_grad()
def boxlist2tensor(boxlists: List[torch.Tensor], tensor_resolution, factor=4) -> torch.Tensor:
    # Input : List[[BoxIndex, X, Y, W, H, CONF]]
    # Output: [SequenceIndex, 1, TensorHeight, TensorWidth]
    ret = np.zeros(
        (len(boxlists), 1, tensor_resolution[1] // factor, tensor_resolution[0] // factor),
        dtype=np.float32)

    for tensor, boxlist in zip(ret, boxlists):
        if 0 == boxlist.nelement():
            continue
        boxlist = boxlist[:, :5].cpu().numpy()
        for (*xyxy, conf) in boxlist:
            intxyxy = [int(element / factor) for element in xyxy]
            (x0, y0, x1, y1) = intxyxy
            tensor[0, y0:y1, x0:x1] += conf
    return torch.from_numpy(ret)


def CASNet(n_inp, n_out=2):
    return nn.Sequential(
            nn.Linear(n_inp, n_inp * 4),
            nn.LeakyReLU(),
            nn.Linear(n_inp * 4, n_inp),
            nn.LeakyReLU(),
            nn.Linear(n_inp, n_out),
            nn.Softmax(dim=1)
        )


class ImagePolicyNet(nn.Module, ABC):
    def __init__(self, n_opt, frozen=False, pretrained=False):
        super(ImagePolicyNet, self).__init__()
        self.backbone = resnet18(pretrained=pretrained)
        self.backbone.fc = nn.Linear(512, n_opt)

        if frozen:
            self.backbone.requires_grad_(False)
            self.backbone.fc.requires_grad_(True)
            self.backbone.eval()
            self.backbone.fc.train()

    def forward(self, x):
        return self.backbone(x)


class SeriesLinearPolicyNet(nn.Module, ABC):
    def __init__(self, n_inp, n_opt):
        super(SeriesLinearPolicyNet, self).__init__()
        self.fc = nn.Linear(n_inp, n_opt)

    def forward(self, x):
        return nn.functional.log_softmax(self.fc(x.view(x.shape[0], -1)), dim=1)


class SimpleBoxMaskCNN(torch.nn.Module, ABC):
    def __init__(self, n_option, n_stack):
        super(SimpleBoxMaskCNN, self).__init__()
        internal = max(1, n_stack // 2)
        self.convs = nn.Sequential(
            nn.Conv2d(n_stack, internal, 5, stride=3),
            nn.LeakyReLU(),
            nn.Conv2d(internal, 3, 5),
            nn.LeakyReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(3312, 128),
            nn.Linear(128, n_option),
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.fc(x.view(x.shape[0], -1))
        return nn.functional.log_softmax(x, dim=1)