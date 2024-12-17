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
    rendered_output = renderer(meshes_world=meshes, cameras=cameras)
    tensor = rendered_output[0, ..., :3].permute(0, 3, 1, 2) # (1, 3, H, W)
    alpha_channel = rendered_output[0, ..., 3]  # Get the alpha channel
    object_masks = (alpha_channel > 0).float()  # Binary mask based on transparency
    return tensor, object_masks


# Save final optimized images in batches
def save_render(renderer, angles, mesh, path):
    os.makedirs(path, exist_ok=True)

    # Prepare batched rotation matrices and translations
    R_list = []
    T_list = []
    for angle, axis in angles:
        R = RotateAxisAngle(angle, axis=axis, device=device).get_matrix()[..., :3, :3]
        R_list.append(R)
        T_list.append(torch.tensor([[0.0, 0.0, 3.0]], device=device))

    # Stack rotation matrices and translations into batches
    R_batch = torch.stack(R_list, dim=0)  # Shape: (n_views, 3, 3)
    T_batch = torch.cat(T_list, dim=0)    # Shape: (n_views, 3)

    # Create batched cameras
    cameras = FoVPerspectiveCameras(R=R_batch, T=T_batch, device=device)

    # Extend the mesh to match the batch size
    batched_meshes = mesh.extend(len(angles))  # Duplicate mesh for all views

    # Render the batch of meshes
    rendered_tensors, _ = render_meshes(renderer, batched_meshes, cameras)

    # Save each rendered image
    for i, tensor in enumerate(rendered_tensors):
        image = tensor_to_image(tensor)  # Convert tensor to PIL image
        image.save(f"{path}/view_{i}.png")

    print(f"Rendered {len(angles)} views and saved to '{path}'")
