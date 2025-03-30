import torch
from PIL import Image
from torchvision import transforms
from models.generator import Generator

def load_model(checkpoint_path):
    # Load the model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = Generator()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def super_resolve_image(model, image_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    
    # Define the transformation to resize the image to 64x64
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Apply the transformation
    input_image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Perform super-resolution
    with torch.no_grad():
        output_image = model(input_image)
    
    # Post-process the output image
    output_image = output_image.squeeze(0)  # Remove batch dimension
    output_image = (output_image + 1) / 2  # Denormalize to [0, 1]
    output_image = transforms.ToPILImage()(output_image.clamp(0, 1))  # Convert to PIL Image
    output_image.save('output_image.png')  # Save the output image
    return output_image