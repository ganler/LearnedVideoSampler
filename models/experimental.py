from abc import ABC

from torch import nn
from torchvision.models import resnet18

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


class SeriesUnlinearPolicyNet(nn.Module, ABC):
    def __init__(self, n_inp, n_opt):
        super(SeriesUnlinearPolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_inp, n_inp),
            nn.LeakyReLU(),
            nn.Linear(n_inp, n_inp // 2),
            nn.LeakyReLU(),
            nn.Linear(n_inp // 2, n_opt)
        )

    def forward(self, x):
        # Input Format: [Batch, Data]
        return nn.functional.log_softmax(self.fc(x.view(x.shape[0], -1)), dim=1)


