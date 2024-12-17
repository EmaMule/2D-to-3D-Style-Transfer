import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
from pytorch3d.transforms import RotateAxisAngle
from pytorch3d.renderer import FoVPerspectiveCameras


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Helper function to blend image with background
def apply_background(rendered_image, mask, background):
    mask = mask.unsqueeze(1)
    return rendered_image * mask + background * (1 - mask)


# Load and preprocess the images
def load_as_tensor(image_path, size=512):

    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

    image = transform(image)[:3, :, :]
    return image.to(device)


# Load the pre-trained VGG19 model
def get_vgg():
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device)
    for param in vgg.parameters():
        param.requires_grad_(False)
    return vgg


# Convert tensor to image for display
def tensor_to_image(tensor):
    image = tensor.clone().detach()
    image = image.squeeze(0)  # Remove batch dimension
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image.cpu())
    return image


# Render the content tensor
def render_meshes(renderer, meshes, cameras):
    tensors = []
    object_masks = []
    for camera in cameras:
        rendered_output = renderer(meshes_world=meshes, cameras=camera)
        tensor = rendered_output[0, ..., :3].permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        alpha_channel = rendered_output[0, ..., 3]  # Get the alpha channel
        object_mask = (alpha_channel > 0).float()  # Binary mask based on transparency
        tensors.append(tensors)
        object_masks.append(object_mask)
    tensors = torch.Tensor(tensors, device = device)
    object_masks = torch.Tensor(object_masks, device = device)
    return tensors, object_masks


# Save final optimized images
def save_render(renderer, meshes, cameras, path):

    os.makedirs(path, exist_ok=True)

    # Render optimized mesh
    tensors, _ = render_meshes(renderer, mesh, cameras)

    for i in range(tensors.shape[0]):
        tensor = tensors[i, ...]
        image = tensor_to_image(tensor)
        image.save(f"{path}/view_{i}.png")