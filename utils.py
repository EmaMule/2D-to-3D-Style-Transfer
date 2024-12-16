import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
from pytorch3d.transforms import RotateAxisAngle
from pytorch3d.renderer import FoVPerspectiveCameras


# Helper function to blend image with background
def apply_background(rendered_image, mask, background):
    return rendered_image * mask + background * (1 - mask)


# Load and preprocess the images
def load_as_tensor(image_path, device, size=512):

    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

    image = transform(image)[:3, :, :]
    return image.to(device)


# Load the pre-trained VGG19 model
def get_vgg(device):
    vgg = models.vgg19(pretrained=True).features.to(device)
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
    rendered_output = renderer(meshes_world=meshes, cameras=cameras)
    tensor = rendered_output[0, ..., :3].permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    alpha_channel = rendered_output[0, ..., 3]  # Get the alpha channel
    object_mask = (alpha_channel > 0).float()  # Binary mask based on transparency
    return tensor, object_mask


# Save final optimized images
def save_render(renderer, angles, mesh, path):

    os.makedirs(path, exist_ok=True)

    for i, (angle, axis) in enumerate(angles):
        R = RotateAxisAngle(angle, axis=axis, device=device).get_matrix()[..., :3, :3]
        T = torch.tensor([[0.0, 0.0, 3.0]], device=device)
        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

        # Render optimized mesh
        tensor, _ = render_meshes(renderer, mesh, cameras)
        image = tensor_to_image(tensor)

        image.save(f"{path}/view_{i}.png")