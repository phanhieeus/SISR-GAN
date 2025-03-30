import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform_hr=None, transform_lr=None):
        self.root_dir = root_dir
        self.transform_hr = transform_hr
        self.transform_lr = transform_lr
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform_hr:
            hr_image = self.transform_hr(image)
        if self.transform_lr:
            lr_image = self.transform_lr(image)            
        return hr_image, lr_image