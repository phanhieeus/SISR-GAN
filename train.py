import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.discriminator import Discriminator
from models.generator import Generator
from utils.loss import PerceptualLoss, DiscriminatorLoss
from utils.dataset import ImageDataset


from tqdm.auto import tqdm

def train(epochs=10, batch_size=1, lr=1e-4):
    
    transform_hr = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform_lr = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = ImageDataset(root_dir='/kaggle/input/celebahq-resized-256x256/celeba_hq_256',
                           transform_hr=transform_hr, transform_lr=transform_lr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    perceptual_loss = PerceptualLoss().to(device)
    discriminator_loss = DiscriminatorLoss().to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.9, 0.999))

    for ep in range(epochs):
        loop = tqdm(dataloader, leave=True, desc=f"Epoch [{ep+1}/{epochs}]")

        total_d_loss = 0.0
        total_g_loss = 0.0
        num_batches = 0

        for hr, lr in loop:
            hr, lr = hr.to(device), lr.to(device)

            # Generate super-resolved images
            sr_imgs = generator(lr)

            # Discriminator predictions
            sr_pred = discriminator(sr_imgs)
            real_pred = discriminator(hr)

            # Compute discriminator loss
            loss_d = discriminator_loss(sr_pred, real_pred)

            # Backprop for discriminator
            optimizer_d.zero_grad()
            loss_d.backward(retain_graph=True)
            
            # Compute generator (perceptual) loss
            loss_g = perceptual_loss(sr_imgs, hr, sr_pred)
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_d.step()
            optimizer_g.step()

            # Accumulate losses
            total_d_loss += loss_d.item()
            total_g_loss += loss_g.item()
            num_batches += 1

            loop.set_postfix(D_loss=loss_d.item(), G_loss=loss_g.item())

        avg_d_loss = total_d_loss / num_batches
        avg_g_loss = total_g_loss / num_batches
        print(f"Epoch {ep+1} finished. Avg D_loss: {avg_d_loss:.4f}, Avg G_loss: {avg_g_loss:.4f}")

        # Save model checkpoints
        if (ep + 1) % 5 == 0:
            torch.save(generator.state_dict(), f'/kaggle/working/generator_epoch_{ep+1}.pth')
            torch.save(discriminator.state_dict(), f'/kaggle/working/discriminator_epoch_{ep+1}.pth')

    print("Training completed.")
