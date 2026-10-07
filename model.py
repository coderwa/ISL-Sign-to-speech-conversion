import torch.nn as nn
from torchvision import models


def build_model(num_classes=26, pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    net = models.resnet18(weights=weights)
    net.fc = nn.Sequential(
        nn.Dropout(0.35),
        nn.Linear(net.fc.in_features, num_classes),
    )
    return net
