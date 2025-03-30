import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
    def forward(self, x):
        return x + self.residual(x)

class Generator(nn.Module):
    def __init__(self, BLOCK=16, scale=2):
        
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.prelu = nn.PReLU()
        self.res_blocks = nn.Sequential(*[ResidualBlock() for _ in range(BLOCK)])
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.Sequential(*[nn.PixelShuffle(scale) for _ in range(2)])
        self.conv3_ = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=4)
        self.conv4 = nn.Conv2d(16, 3, kernel_size=9, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu(x)
        res = self.res_blocks(x)
        res = self.conv2(res)
        res = self.bn(res)
        x = x + res
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        x = self.conv3_(x)
        x = self.prelu(x)
        x = self.conv4(x)       
        return x

    
