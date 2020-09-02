import torch
from torch import nn
from torchvision.models import resnet18

'''
Input signal can be
=> current image.
=> historical input data. 
'''

class ImageEncoder(nn.Module):
    def __init__(self, n_out=64):
        self.backbone = resnet18(pretrained=True)
        self.embedding = nn.Linear(1000, n_out)

    def forward(self, x):
        return self.embedding(self.backbone(x))

def encode_box(pred: torch.Tensor, box_max_size=64):
    ret = torch.zeros((len(pred), box_max_size * 5))
    for i, det in enumerate(pred):
        det: torch.Tensor
        ret[0:det.view(-1).shape[0]] = det.view(1, -1)
    return ret

class Box3Encoder(nn.Module):
    def __init__(self, n_in=64*5, n_hidden=256, n_out=64):
        self.rnn = nn.LSTM(n_in, n_hidden, bidirectional=True, batch_first=True)
        self.embedding = nn.Linear(n_hidden * 2, n_out)

    def forward(self, inp):
        recurrent, _ = self.rnn(inp)
        out = self.embedding(recurrent[:, -1, :])
        return out


class SamplerBackbone(torch.nn.Module):
    def __init__(self, n_option, n_in=64*5, n_hidden=256, n_embed=64):
        super(SamplerBackbone, self).__init__()
        self.image_encoder = ImageEncoder(n_embed)
        self.box_encoder = Box3Encoder(n_in, n_hidden, n_embed)

        self.img_mul = nn.Linear(n_embed, n_embed)
        self.box_mul = nn.Linear(n_embed, n_embed)

        self.post = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.LeakyReLU(),
            nn.Linear(n_embed, n_option)
        )

    def forward(self, im, boxes):
        l = self.image_encoder(im) * self.img_mul
        r = self.box_encoder(boxes) * self.box_mul

        x = l + r
        x = self.post(x)
        print(x.shape)
        return nn.functional.log_softmax(x, dim=2)