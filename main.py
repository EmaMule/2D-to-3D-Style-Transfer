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
style_tensor = load_as_tensor(style_image_path, device, size=size).unsqueeze(0).to(device)

# Load VGG model
vgg = get_vgg(device=device)

# Define angles for viewpoints
angles_x = torch.linspace(0, 270, n_views)  # X-axis rotation
angles_y = torch.linspace(90, 270, n_views - 2)  # Y-axis rotation
angles = [(angle.item(), "X") for angle in angles_x] + [(angle.item(), "Y") for angle in angles_y]

# Initialize texture optimization
current_cow_mesh = content_cow_mesh.clone()
texture_map = current_cow_mesh.textures.maps_padded()
texture_map.requires_grad = True
optimizer = torch.optim.Adam([texture_map], lr=learning_rate)

# Render and optimize for each viewpoint
for i, (angle, axis) in enumerate(angles):
    # Set up camera rotation
    R = RotateAxisAngle(angle, axis=axis, device=device).get_matrix()[..., :3, :3]
    T = torch.tensor([[0.0, 0.0, 3.0]], device=device)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

    # Render the content image
    content_tensor, content_object_mask = render_meshes(renderer, content_cow_mesh, cameras)

    if use_background == 2:
        content_tensor = apply_background(content_tensor, content_object_mask, style_tensor)

    # Render the current texture
    current_tensor, current_object_mask = render_meshes(renderer, current_cow_mesh, cameras)

    if use_background in [1,2]:
        current_tensor = apply_background(current_tensor, current_object_mask, style_tensor)

    # Perform style transfer
    applied_style_tensor = style_transfer(current_tensor, content_tensor, style_tensor, vgg, steps=n_style_transfer_steps,
                                  style_weight=style_weight, content_weight=content_weight)

    # Save the styled image
    applied_style_image = tensor_to_image(applied_style_tensor)
    applied_style_image.save(output_path+f"/2d_style_transfer/view_{i}.png")

    # Optimize the texture to match the styled image
    for step in range(n_mse_steps):
        optimizer.zero_grad()
        current_cow_mesh.textures = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_map)

        rendered_tensor, object_mask = render_meshes(renderer, current_cow_mesh, cameras)

        # Compute loss with the mask applied
        masked_rendered = rendered_tensor * object_mask
        masked_target = applied_style_tensor * object_mask
        loss = torch.nn.functional.mse_loss(masked_rendered, masked_target)

        # Backpropagation
        loss.backward()
        optimizer.step()

        print(f"View {i}, Step {step}, Loss: {loss.item()}")
    
    save_render(renderer, angles, current_cow_mesh, output_path+f"/iteration_{i}")


# Save final optimized images
optimized_cow_mesh = current_cow_mesh
save_render(renderer, angles, optimized_cow_mesh, output_path+"/final_render")