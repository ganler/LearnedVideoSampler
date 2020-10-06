from abc import ABC

import numpy as np
import torch
from torch import nn
from torchvision.models import mobilenet_v2, resnet18, vgg11, mnasnet1_3
from typing import List
import torch.nn.functional as F
import cv2

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


class ImageEncoder(nn.Module, ABC):
    def __init__(self, n_out=64, frozen=True):
        super(ImageEncoder, self).__init__()
        self.backbone = resnet18(pretrained=True)

        if frozen:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        self.embedding = nn.Linear(1000, n_out)

    def forward(self, x):
        return self.embedding(self.backbone(x))


class BBoxListEncoder(nn.Module, ABC):
    def __init__(self, n_hidden=64, n_out=64):
        super(BBoxListEncoder, self).__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(1, 4, 5),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(4, 4, 5),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(4, 8, 3),
            nn.LeakyReLU(inplace=True)
        )
        self.rnn = nn.LSTM(8, n_hidden, bidirectional=True, batch_first=True)
        self.embedding = nn.Sequential(
            nn.Linear(n_hidden * 2, n_out)
        )

    def forward(self, x: torch.Tensor):
        # [BatchDim, SequenceDim, 1, W, H]

        has_batch = len(x.shape) == 5
        if has_batch:
            N, S, C, H, W = x.shape
            x = x.view(-1, C, H, W)
        x = self.bottleneck(x)
        # [BatchDim, SequenceDim, 64, ?, ?]
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        if has_batch:
            x = x.view(N, S, -1)
        else:
            x = x.unsqueeze(0)
        # [BatchDim, SequenceDim, 64]
        x, _ = self.rnn(x)
        return self.embedding(x[:, -1, :])


class SamplerBackbone(torch.nn.Module, ABC):
    def __init__(self, n_option, n_hidden=256, n_embed=64):
        super(SamplerBackbone, self).__init__()
        self.image_encoder = ImageEncoder(n_embed)
        self.box_encoder = BBoxListEncoder(n_hidden=n_hidden, n_out=n_embed)
        self.post = nn.Linear(n_embed, n_option)

    def forward(self, im, boxes):
        left = self.image_encoder(im)
        right = self.box_encoder(boxes)

        x = left + right
        x = self.post(x)
        return nn.functional.log_softmax(x, dim=1)
