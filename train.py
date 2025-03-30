import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.discriminator import Discriminator
from models.generator import Generator
from utils.loss import PerceptualLoss, DiscriminatorLoss
from utils.dataset import ImageDataset


def train(epochs=10, batch_size=16, lr=1e-4):
    
    transform_hr = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform_lr = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


    dataset = ImageDataset(root_dir='data/train', transform_hr=transform_hr, transform_lr=transform_lr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    perceptual_loss = PerceptualLoss().to(device)
    discriminator_loss = DiscriminatorLoss().to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.9, 0.999))



    for ep in range(epochs):
        for hr, lr in dataloader:
            hr, lr = hr.to(device), lr.to(device)
            # Generate super-resolved images
            sr_imgs = generator(lr)

            # Discriminator predictions
            sr_pred = discriminator(sr_imgs)
            real_pred = discriminator(hr)

            # Compute losses
            loss_g = perceptual_loss(sr_imgs, hr, sr_pred)
            loss_d = discriminator_loss(sr_pred, real_pred)

            # Backpropagation
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

        print(f'Epoch [{ep+1}/{epochs}], Loss G: {loss_g.item()}, Loss D: {loss_d.item()}')
        # Save model checkpoints
        if (ep + 1) % 5 == 0:
            torch.save(generator.state_dict(), f'checkpoints/generator_epoch_{ep+1}.pth')
            torch.save(discriminator.state_dict(), f'checkpoints/discriminator_epoch_{ep+1}.pth')


if __name__ == '__main__':
    train()
