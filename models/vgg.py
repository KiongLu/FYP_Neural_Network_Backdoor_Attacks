"""
VGG-16 with batch norm for 32x32 images (Simonyan & Zisserman, 2015). The five max-pools take the
input down to 1x1x512, followed by one linear layer. Used as a second architecture.
"""

import torch.nn as nn

VGG16_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]


class VGG(nn.Module):
    def __init__(self, cfg=VGG16_CFG, num_classes=10):
        super().__init__()
        layers, in_ch = [], 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [nn.Conv2d(in_ch, v, 3, padding=1, bias=False), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_ch = v
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_features=False):
        feat = self.features(x).flatten(1)
        logits = self.fc(feat)
        if return_features:
            return logits, feat
        return logits


def VGG16(num_classes=10):
    return VGG(VGG16_CFG, num_classes=num_classes)
