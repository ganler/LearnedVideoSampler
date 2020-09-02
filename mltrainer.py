from torchvision.models import resnet18
import torch



print(model(torch.zeros((1, 3, 112, 112))).shape)