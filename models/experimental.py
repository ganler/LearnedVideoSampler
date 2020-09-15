from abc import ABC

from torch import nn
from torchvision.models import resnet18

class ImagePolicyNet(nn.Module, ABC):
    def __init__(self, n_opt, frozen=False, pretrained=True):
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