import torch
from torch.nn import functional as F
from torchvision import transforms, models
from PIL import Image
import os
from pytorch3d.transforms import RotateAxisAngle
from pytorch3d.renderer import FoVPerspectiveCameras, TexturesUV
from pytorch3d.structures import Meshes
from pytorch3d.renderer.cameras import look_at_view_transform
import random

from style_transfer import *

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Helper function to blend image with background
def apply_background(tensors, masks, background_type = 'noise', background = None):

    if background_type == 'noise':
        backgrounds = torch.rand(tensors.shape, device = device)
        return tensors * masks + backgrounds * (1 - masks)

    elif background_type == 'style':
        backgrounds = background
        return tensors * masks + background * (1 - masks)

    elif background_type == 'white':
        return tensors


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
    # Extract components of the mesh
    texture_map = mesh.textures.maps_padded()
    verts_uvs = mesh.textures.verts_uvs_padded()
    faces_uvs = mesh.textures.faces_uvs_padded()

    # Extract geometry
    verts = mesh.verts_padded()  # Access the vertices
    faces = mesh.faces_padded()  # Access the faces

    # Finalize the texture (ensure colors in range (0,1) )
    final_texture_map = finalize_tensor(texture_map)

    # Build the final textures
    current_textures = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=final_texture_map)

    # Build the final mesh
    final_mesh = Meshes(verts=verts, faces=faces, textures=current_textures)

    return final_mesh


def finalize_tensor(tensor):
    final_tensor = torch.clamp(tensor, 0.0, 1.0).detach()
    return final_tensor


def build_fixed_cameras(n_views, dist=3.0, shuffle = True):

    # Define angles for viewpoints
    x_views = (n_views // 2)
    y_views = n_views - x_views

    angles_x = torch.linspace(0, 315, x_views)  # Equally spaced angles for X-axis
    angles_y = torch.linspace(45, 315, y_views)  # Equally spaced angles for Y-axis
        
    # CHANGE ANGLE RANGES?
    
    angles = [(angle.item(), "X") for angle in angles_x] + [(angle.item(), "Y") for angle in angles_y]

    # Shuffle angles instead of sampling cameras
    if shuffle:
        random.shuffle(angles)

    # Define camera list
    R_list = []
    T_list = []
    for angle, axis in angles:
        R = RotateAxisAngle(angle, axis=axis, device=device).get_matrix()[..., :3, :3].squeeze(0)
        R_list.append(R)
        # CHANGE DISTANCE?
        T_list.append(torch.tensor([0.0, 0.0, dist], device=device))
    R_list = torch.stack(R_list, dim=0)  # (n_views, 3, 3)
    T_list = torch.stack(T_list, dim=0).squeeze(1)  # (n_views, 3)

    cameras_list = FoVPerspectiveCameras(R=R_list, T=T_list, device=device)

    return cameras_list


def build_random_cameras(n_views, dist=2.10):

    cos_elevs = torch.rand(n_views) * 2 - 1
    elevs = torch.acos(cos_elevs) * 180 / torch.pi - 90 
    
    azims = torch.rand(n_views) * 360 - 180

    R_list, T_list = look_at_view_transform(
        dist = dist,
        elev = elevs,
        azim = azims,
        at=((0, 0.10, 0.25),)
    )

    cameras_list = FoVPerspectiveCameras(R=R_list, T=T_list, device=device)

    return cameras_list


def setup_optimizations(optimization_target, mesh, lr):

    optimizable_mesh = mesh.clone()

    texture_map = optimizable_mesh.textures.maps_padded()
    verts = optimizable_mesh.verts_packed()
    faces = optimizable_mesh.faces_packed()
    verts_uvs = optimizable_mesh.textures.verts_uvs_padded()
    faces_uvs = optimizable_mesh.textures.faces_uvs_padded()
    
    if optimization_target == 'texture':
        texture_map.requires_grad_(True)
        optimizer = torch.optim.Adam([texture_map], lr=lr)

    elif optimization_target == 'mesh':
        verts.requires_grad_(True)
        # optimizing verts_uv doesn't get good results
        optimizer = torch.optim.Adam([verts], lr=lr)

    elif optimization_target == 'both':
        texture_map.requires_grad_(True)
        verts.requires_grad_(True)
        optimizer = torch.optim.Adam([verts, texture_map], lr = lr)
    
    return {'optimizable_mesh': optimizable_mesh,
            'optimizer': optimizer,
            'texture_map': texture_map,
            'verts': verts,
            'faces': faces,
            'verts_uvs': verts_uvs,
            'faces_uvs': faces_uvs
            }


def build_mesh(verts_uvs, faces_uvs, texture_map, verts, faces):
    textures = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_map)
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    return mesh