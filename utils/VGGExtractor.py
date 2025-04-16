import torch.nn as nn
from torchvision import models


class VGGExtractor(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.vgg19 = models.vgg19(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(self.vgg19.children())[:layers])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    def forward(self, x):
        return self.feature_extractor(x)