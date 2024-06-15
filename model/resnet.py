import torch
import torch.nn as nn
from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck
from torchvision import models


class PCRResNet(nn.Module):

    def __init__(self, output_dim=1024, pretrained=True, *args, **kwargs):
        super(PCRResNet, self).__init__()
        ref = models.resnet50(pretrained=pretrained)

        self.conv0_1 = nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=3)
        self.bn0_1 = nn.BatchNorm2d(16)
        self.conv0_2 = nn.Conv2d(16, 16, kernel_size=7, stride=1, padding=3)
        self.bn0_2 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.block0 = nn.Sequential(self.conv0_1, self.bn0_1, self.conv0_2, self.bn0_2, self.relu)
        self.block1 = nn.Sequential(self.conv1, ref.bn1, ref.relu)
        self.block2 = nn.Sequential(ref.maxpool, ref.layer1)
        self.block3 = ref.layer2
        self.block4 = ref.layer3
        self.block5 = ref.layer4
        self.avgpool = ref.avgpool
        self.fc = nn.Linear(2048, output_dim)

    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        x6 = self.avgpool(x5)
        x6 = torch.flatten(x6, 1)
        x6 = self.fc(x6)

        return {
            "geometry_feature": [x0, x1, x2],
            "latent_code": x6,
            "all": [x0, x1, x2, x3, x4, x5, x6]
        }
