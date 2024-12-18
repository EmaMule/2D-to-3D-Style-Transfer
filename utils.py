import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
from pytorch3d.transforms import RotateAxisAngle
from pytorch3d.renderer import FoVPerspectiveCameras, TexturesUV

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Helper function to blend image with background
def apply_background(tensors, masks, backgrounds):
    return tensors * masks + backgrounds * (1 - masks)


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
        tensor = rendered_output[0, ..., :3].permute(2, 0, 1) # (3, H, W)
        alpha_channel = rendered_output[0, ..., 3]  # Get the alpha channel
        object_mask = (alpha_channel > 0).float().unsqueeze(0)  # Binary mask based on transparency
        tensors.append(tensor)
        object_masks.append(object_mask)
    tensors = torch.stack(tensors, dim = 0) # (BATCH, 3, H, W)
    object_masks = torch.stack(object_masks, dim = 0)
    return tensors, object_masks


# Save final optimized images
def save_render(renderer, meshes, cameras, path):

    os.makedirs(path, exist_ok=True)

    # Render optimized mesh
    tensors, _ = render_meshes(renderer, meshes, cameras)

    for i in range(tensors.shape[0]):
        tensor = tensors[i, ...]
        image = tensor_to_image(tensor)
        image.save(f"{path}/view_{i}.png")


def finalize_mesh(mesh):

    # extract components
    texture_map = mesh.textures.maps_padded()
    verts_uvs = mesh.textures.verts_uvs_padded()
    faces_uvs = mesh.textures.faces_uvs_padded()

    # finalize texture
    final_texture_map = torch.clamp(texture_map, 0.0, 1.0)

    # finalize geometry

    # build final mesh
    final_mesh = mesh.clone()
    final_mesh.textures = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=final_texture_map)

    return final_mesh


def build_cameras(n_views, shuffle = True):

    # Define angles for viewpoints
    x_views = (n_views // 2)
    y_views = n_views - x_views
    angles_x = torch.linspace(0, 315, x_views)  # X-axis rotation
    angles_y = torch.linspace(45, 315, y_views)  # Y-axis rotation
    
    angles = [(angle.item(), "X") for angle in angles_x] + [(angle.item(), "Y") for angle in angles_y]

    # Shuffle angles instead of sampling cameras
    random.shuffle(angles)

    # Define camera list
    R_list = []
    T_list = []
    for angle, axis in angles:
        R = RotateAxisAngle(angle, axis=axis, device=device).get_matrix()[..., :3, :3].squeeze(0)
        R_list.append(R)
        T_list.append(torch.tensor([0.0, 0.0, 3.0], device=device))
    R_list = torch.stack(R_list, dim=0)  # (n_views, 3, 3)
    T_list = torch.stack(T_list, dim=0).squeeze(1)  # (n_views, 3)

    cameras_list = FoVPerspectiveCameras(R=R_list, T=T_list, device=device)

    return cameras_list