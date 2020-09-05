from abc import ABC

import numpy as np
import torch
from torch import nn
from torchvision.models import mobilenet_v2, resnet18, vgg11, mnasnet1_3
from typing import List
import torch.nn.functional as F

'''
Input signal can be
=> current image.
=> historical input data. 
'''


@torch.no_grad()
def boxlist2tensor(boxlists: List[torch.Tensor], tensor_resolution, factor=1) -> torch.Tensor:
    # Input : List[[BoxIndex, X, Y, W, H, CONF]]
    # Output: [SequenceIndex, 1, TensorHeight, TensorWidth]
    ret = np.zeros((len(boxlists), 1, tensor_resolution[0] // factor, tensor_resolution[1] // factor), dtype=np.float32)

    for tensor, boxlist in zip(ret, boxlists):
        if 0 == boxlist.nelement():
            continue
        boxlist = boxlist[:, :5].cpu().numpy()
        for (*xyxy, conf) in boxlist:
            intxyxy = [int(element / factor) for element in xyxy]
            (x0, y0, x1, y1) = intxyxy
            tensor[0, x0:x1, y0:y1] += conf
    return torch.from_numpy(ret).float()


def make_bottleneck(input_channel=1):
    return nn.Sequential(
        nn.Conv2d(input_channel, input_channel * 4, 5),
        nn.ReLU(inplace=True),
        nn.Conv2d(input_channel * 4, input_channel * 8, 3),
        nn.ReLU(inplace=True)
    )

class ImageEncoder(nn.Module, ABC):
    def __init__(self, n_out=64):
        super(ImageEncoder, self).__init__()
        self.backbone = resnet18(pretrained=False)

        # self.diynet = nn.Sequential(
        #     make_bottleneck(3),
        #     nn.MaxPool2d(3),
        #     make_bottleneck(3 * 8)
        # )
        # self.diyembedding = nn.Linear(192, n_out)

        self.embedding = nn.Linear(1000, n_out)

    def forward(self, x):
        return self.embedding(self.backbone(x))


class BBoxListEncoder(nn.Module, ABC):
    def __init__(self, n_hidden=256, n_out=64):
        super(BBoxListEncoder, self).__init__()

        self.bottleneck0 = make_bottleneck(1)
        self.bottleneck1 = make_bottleneck(8)  # 8 x 8
        self.rnn = nn.LSTM(64, n_hidden, bidirectional=True, batch_first=True)
        self.embedding = nn.Linear(n_hidden * 2, n_out)

    def forward(self, x: torch.Tensor):
        # [BatchDim, SequenceDim, 1, W, H]

        has_batch = len(x.shape) == 5
        if has_batch:
            N, S, C, H, W = x.shape
            x = x.view(-1, C, H, W)
        x = self.bottleneck0(x)
        x = self.bottleneck1(x)
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

        self.img_mul = nn.Linear(n_embed, n_embed)
        self.box_mul = nn.Linear(n_embed, n_embed)

        self.post = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(n_embed, n_option)
        )

    def forward(self, im, boxes):
        left = self.img_mul(self.image_encoder(im))
        right = self.box_mul(self.box_encoder(boxes))

        x = left + right
        x = self.post(x)
        return nn.functional.log_softmax(x, dim=1)
