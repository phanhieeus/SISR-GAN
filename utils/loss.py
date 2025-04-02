import torch
import torch.nn as nn
from torchvision import models
from VGGExtractor import VGGExtractor



class ContentLoss(nn.Module):

    def __init__(self, layers=9):
        super().__init__()
        self.vgg_extractor = VGGExtractor(layers).eval()
        self.criterion = nn.MSELoss()
        
    def forward(self, sr, hr):
        sr_features = self.vgg_extractor(sr)
        hr_features = self.vgg_extractor(hr)
        loss = self.criterion(sr_features, hr_features)
        return loss
    
class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
    def forward(self, sr_pred):
        loss = self.criterion(sr_pred, torch.ones_like(sr_pred))
        return loss

class PerceptualLoss(nn.Module):
    def __init__(self, layers=9):
        super().__init__()
        self.content_loss = ContentLoss(layers)
        self.adversarial_loss = AdversarialLoss()
    def forward(self, sr, hr, sr_pred):
        content_loss = self.content_loss(sr, hr)
        adversarial_loss = self.adversarial_loss(sr_pred)
        return content_loss + 1e-3 * adversarial_loss
        
class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
    def forward(self, sr_pred, real_pred):
        loss = self.criterion(sr_pred, torch.zeros_like(sr_pred)) + self.criterion(real_pred, torch.ones_like(real_pred))
        return loss / 2