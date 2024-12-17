import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
from PIL import Image

# Import style transfer utilities
from style_transfer import style_transfer
from utils import apply_background, get_vgg, load_as_tensor, tensor_to_image, render_meshes, save_render

from torchvision import transforms

# Import PyTorch3D utilities
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV, FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader, PointLights, AmbientLights
from pytorch3d.transforms import RotateAxisAngle

import argparse

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--n_views", default=4, type=int, help="Number of views considered by the renderer")
parser.add_argument("--n_mse_steps", default=100, type=int, help="Number of steps for MSE optimization")
parser.add_argument("--n_style_transfer_steps", default=3000, type=int, help="Number of steps for style transfer")
parser.add_argument("--use_background", default=0, type=int, help="0: No background, 1: Background on rendered image, 2: Background on both")
parser.add_argument("--obj_path", default="./objects/cow_mesh/cow.obj", type=str, help="Path to the object")
parser.add_argument("--style_path", default="./imgs/Style_1.jpg", type=str, help="Path to the style image")
parser.add_argument("--style_weight", default=1e6, type=float, help="Weight of the style loss")
parser.add_argument("--content_weight", default=1.0, type=float, help="Weight of the content loss")
parser.add_argument("--size", default=512, type=int, help="Dimension of the images")
parser.add_argument("--output_path", default="/content/output", type=str, help="Output folder path")
args = parser.parse_args()

# Parse arguments
cow_obj_path = args.obj_path
style_image_path = args.style_path
n_views = args.n_views
n_mse_steps = args.n_mse_steps
n_style_transfer_steps = args.n_style_transfer_steps
use_background = args.use_background
content_weight = args.content_weight
style_weight = args.style_weight
size = args.size
output_path = args.output_path

# Other Parameters
learning_rate = 0.01
style_transfer_lr = 0.003
batch_size = n_views

# Create output folder
os.makedirs(output_path, exist_ok=True)
os.makedirs(output_path+"/2d_style_transfer", exist_ok=True)

# Set device (use GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the cow mesh
verts, faces, aux = load_obj(cow_obj_path)
verts_uvs = aux.verts_uvs[None, ...].to(device)  # (1, V, 2)
faces_uvs = faces.textures_idx[None, ...].to(device)  # (1, F, 3)
texture_image = list(aux.texture_images.values())[0][None, ...].to(device)  # (1, H, W, 3)

# Initialize textures and mesh
original_textures = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)
content_cow_mesh = Meshes(verts=[verts.to(device)], faces=[faces.verts_idx.to(device)], textures=original_textures)

content_cow_meshes = content_cow_mesh.extend(batch_size) #extends the mesh of the content to batch_size (num views)

#Remains unchanged!

# Camera, rasterization, and lighting settings
cameras = FoVPerspectiveCameras(device=device)
raster_settings = RasterizationSettings(image_size=size, blur_radius=0.0, faces_per_pixel=1)
# try AmbientLights instead!
lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

# Create a renderer
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
)

# Load style image
style_tensor = load_as_tensor(style_image_path, size=size).unsqueeze(0).to(device)

# Load VGG model
vgg = get_vgg()

# Define angles for viewpoints
angles_x = torch.linspace(0, 270, n_views)  # X-axis rotation
angles_y = torch.linspace(90, 270, n_views - 2)  # Y-axis rotation
angles = [(angle.item(), "X") for angle in angles_x] + [(angle.item(), "Y") for angle in angles_y]

# Initialize texture optimization --> work in batches!
current_cow_meshes = content_cow_meshes.clone()
texture_maps = current_cow_meshes.textures.maps_padded() #are they all?

texture_maps.requires_grad = True
optimizer = torch.optim.Adam([texture_maps], lr=learning_rate)


# Batch Rotation Matrices
R_list = []
T_list = []
for angle, axis in angles:
    R = RotateAxisAngle(angle, axis=axis, device=device).get_matrix()[..., :3, :3]
    R_list.append(R)
    T_list.append(torch.tensor([[0.0, 0.0, 3.0]], device=device))
R_batch = torch.stack(R_list, dim=0)
T_batch = torch.stack(T_list, dim=0).squeeze(1)

# Batch Cameras
cameras = FoVPerspectiveCameras(R=R_batch, T=T_batch, device=device)

# Render content and current images for all views
content_tensors, content_masks = render_meshes(renderer, content_cow_meshes, cameras)
current_tensors, current_masks = render_meshes(renderer, current_cow_meshes, cameras)

# Apply background if needed
if use_background in [1, 2]:
    content_tensors = apply_background(content_tensors, content_masks, style_tensor)
    current_tensors = apply_background(current_tensors, current_masks, style_tensor)

# Perform batch style transfer
styled_tensors = style_transfer(current_tensors, content_tensors, style_tensor, vgg, steps=n_style_transfer_steps,
                                style_weight=style_weight, content_weight=content_weight, lr=style_transfer_lr)

# Save styled images
for i, styled_tensor in enumerate(styled_tensors):
    applied_style_image = tensor_to_image(styled_tensor)
    applied_style_image.save(output_path+f"/2d_style_transfer/view_{i}.png")

# Optimize the texture map in batches
for step in range(n_mse_steps):
    optimizer.zero_grad()
    current_cow_meshes.textures = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_maps)
    rendered_tensors, object_masks = render_meshes(renderer, current_cow_meshes, cameras)
    
    # Compute masked MSE loss
    masked_rendered = rendered_tensors * object_masks
    masked_target = styled_tensors * object_masks
    loss = torch.nn.functional.mse_loss(masked_rendered, masked_target)

    # Backpropagation
    loss.backward()
    optimizer.step()

    print(f"Step {step}, Loss: {loss.item()}")

# Save final optimized render
save_render(renderer, angles, current_cow_meshes, output_path+"/final_render")