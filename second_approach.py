import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import math
import random

# Import style transfer utilities
from style_transfer import compute_perceptual_loss
from utils import apply_background, get_vgg, load_as_tensor, tensor_to_image, render_meshes, save_render, finalize_mesh, build_cameras

from torchvision import transforms

# Import PyTorch3D utilities
from pytorch3d.io import load_obj
from pytorch3d.io import IO
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV, FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader, PointLights, AmbientLights
from pytorch3d.transforms import RotateAxisAngle

import argparse

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--n_views", default=6, type=int, help="Number of views considered by the renderer")
parser.add_argument("--epochs", default=3000, type=int, help="Number of epochs for style transfer")
parser.add_argument("--obj_path", default="./objects/cow_mesh/cow.obj", type=str, help="Path to the object")
parser.add_argument("--style_path", default="./imgs/Style_1.jpg", type=str, help="Path to the style image")
parser.add_argument("--style_weight", default=1e6, type=float, help="Weight of the style loss")
parser.add_argument("--content_weight", default=1.0, type=float, help="Weight of the content loss")
parser.add_argument("--size", default=768, type=int, help="Dimension of the images") # (default value is texture resolution)
parser.add_argument("--output_path", default="/content/output_second", type=str, help="Output folder path")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
parser.add_argument("--content_background", default='white', type=str, help="Type of background for the content image")
parser.add_argument("--current_background", default='white', type=str, help="Type of background for the current image")
parser.add_argument("--lr", default=0.01, type=float, help="Style Transfer Learning Rate")
args = parser.parse_args()

# Parse arguments
cow_obj_path = args.obj_path
style_image_path = args.style_path
n_views = args.n_views
epochs = args.epochs
content_weight = args.content_weight
style_weight = args.style_weight
size = args.size
output_path = args.output_path
batch_size = args.batch_size
content_background = args.content_background
current_background = args.current_background
lr = args.lr

assert content_background in ['noise', 'style', 'white']
assert current_background in ['noise', 'style', 'white']

# Create output folder
os.makedirs(output_path, exist_ok=True)
os.makedirs(output_path+"/current_images", exist_ok=True)

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
lights = AmbientLights(device = device)

# Create a renderer
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
)

# Load VGG model
vgg = get_vgg()

# Build cameras
cameras_list = build_cameras(n_views)
# CAMERAS SHOULD BE RANDOM INSTEAD OF FIXED AND BATCHED?

# Initialize texture optimization
current_cow_mesh = content_cow_mesh.clone()
texture_map = current_cow_mesh.textures.maps_padded()
texture_map.requires_grad = True
optimizer = torch.optim.Adam([texture_map], lr=lr)


# USE TWO CURRENT: ONE ALWAYS WITH STYLE AND ONE WITH THE SAME AS CONTENT

for epoch in tqdm(range(epochs)):

    total_loss = 0

    for i in range(math.ceil(n_views / batch_size)):

        optimizer.zero_grad()
        batch_start = i*batch_size
        batch_end = min((i+1)*batch_size, n_views)
        current_batch_size = batch_end - batch_start
        
        # sample cameras (shuffling is done in the angles, they can be taken in order)
        batch_indexes = list(range(batch_start, batch_end))
        batch_cameras = [cameras_list[idx] for idx in batch_indexes]

        # Load style image
        style_tensors = load_as_tensor(style_image_path, size=size).repeat(current_batch_size, 1, 1, 1).to(device)

        # Render content images for all views
        content_tensors, content_masks = render_meshes(renderer, content_cow_mesh, batch_cameras)
        if content_background == 'noise':
            content_tensors = apply_background(content_tensors, content_masks, torch.rand(style_tensors.shape, device = device))
        elif content_background == 'style':
            content_tensors = apply_background(content_tensors, content_masks, style_tensors)
        # else content_background == 'white' does nothing

        # Render current images for all views (only if used)
        current_tensors, current_masks = render_meshes(renderer, current_cow_mesh, batch_cameras)
        if current_background == 'noise':
            current_tensors = apply_background(current_tensors, current_masks, torch.rand(style_tensors.shape, device=device))
        elif current_background == 'style':
            current_tensors = apply_background(current_tensors, current_masks, style_tensors)
        # else current_background == 'white' does nothing

        loss = compute_perceptual_loss(current_tensors, content_tensors, style_tensors, vgg, style_weight=style_weight, content_weight=content_weight)

        # Save styled images
        for j, current_tensor in enumerate(current_tensors):
            applied_style_image = tensor_to_image(current_tensor)
            applied_style_image.save(output_path + f"/current_images/view_{i*batch_size+j}.png")

        # Backpropagation
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

# Ensure texture values are in the correct range
final_cow_mesh = finalize_mesh(current_cow_mesh)

# Save final optimized images
save_render(renderer, final_cow_mesh, cameras_list, output_path+"/final_render")
IO().save_mesh(final_cow_mesh, output_path+"/final.obj")