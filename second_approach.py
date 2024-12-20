import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import math
import random
import argparse

# Import style transfer utilities
from style_transfer import *
from utils import *

from torchvision import transforms

# Import PyTorch3D utilities
from pytorch3d.io import load_obj, IO
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV, FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader, AmbientLights

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
parser.add_argument("--content_background", default='white', type=str, choices=['noise', 'style', 'white'], help="Type of background for the content image")
parser.add_argument("--current_background", default='white', type=str, choices=['noise', 'style', 'white'], help="Type of background for the current image")
parser.add_argument("--lr", default=0.01, type=float, help="Style Transfer Learning Rate")
parser.add_argument("--randomize_views", type=bool, default=True, help="Whether or not to randomize views")
parser.add_argument("--optimization_target", type=str, choices=['texture', 'mesh', 'both'], default="texture", help="Decide what to optimize")
parser.add_argument("--main_loss_weight", type=float, default=3.0, help="Weight of the main computed loss (i.e., perceptual)")
parser.add_argument("--mesh_edge_loss_weight", type=float, default=1.0, help="Weight of edge loss (enforces admissible weights for the edges)")
parser.add_argument("--mesh_laplacian_smoothing_weight", type=float, default=1.0, help="Weight of smoothing (smooth surface)")
parser.add_argument("--mesh_normal_consistency_weight", type=float, default=1.0, help="Weight of normal consistency")
parser.add_argument("--mesh_verts_weight", type=float, default=1.0, help="Mesh verts (uvs and not uvs) regularization weight")

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
randomize_views=args.randomize_views
optimization_target = args.optimization_target

loss_weights = {
    'mesh_edge_loss_weight': args.mesh_edge_loss_weight,
    'mesh_laplacian_smoothing_weight': args.mesh_laplacian_smoothing_weight,
    'mesh_normal_consistency_weight': args.mesh_normal_consistency_weight,
    'mesh_verts_weight': args.mesh_verts_weight,
    'main_loss_weight': args.main_loss_weight
}

# Create output folder
os.makedirs(output_path, exist_ok=True)
os.makedirs(output_path+"/current_images", exist_ok=True)

# Set device (use GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the cow mesh
original_verts, original_faces, aux = load_obj(cow_obj_path)

original_verts_uvs = aux.verts_uvs[None, ...].to(device)  # (1, V, 2)
original_faces_uvs = original_faces.textures_idx[None, ...].to(device)  # (1, F, 3)
original_faces = original_faces.verts_idx.to(device)
texture_image = list(aux.texture_images.values())[0][None, ...].to(device)  # (1, H, W, 3)

original_verts = original_verts.to(device)

# Initialize content textures and mesh
content_cow_mesh = build_mesh(original_verts_uvs, original_faces_uvs, texture_image, original_verts, original_faces)

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
if randomize_views:
    cameras_list = build_random_cameras(n_views)
else:
    cameras_list = build_fixed_cameras(n_views)

#initialize optimization based on the target (it returns the mesh to optimize etc.)
out = setup_optimizations(optimization_target, content_cow_mesh, lr)

#retrieve outputs (done like this for clarity)
current_cow_mesh = out['optimizable_mesh']
optimizer = out['optimizer']
texture_map = out['texture_map']
verts = out['verts']
faces = out['faces']
verts_uvs = out['verts_uvs']
faces_uvs = out['faces_uvs']

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

        #done because pytorch otherwise cries
        current_cow_mesh = build_mesh(verts_uvs, faces_uvs, texture_map, verts, faces)

        current_tensors, current_masks = render_meshes(renderer, current_cow_mesh, batch_cameras)
        if current_background == 'noise':
            current_tensors = apply_background(current_tensors, current_masks, torch.rand(style_tensors.shape, device=device))
        elif current_background == 'style':
            current_tensors = apply_background(current_tensors, current_masks, style_tensors)
        # else current_background == 'white' does nothing

        loss = compute_second_approach_loss(
            current = current_tensors,
            content = content_tensors,
            style = style_tensors,
            model = vgg,
            style_weight = style_weight,
            content_weight = content_weight,
            verts = verts,
            target_verts = original_verts,
            verts_uvs = verts_uvs,
            target_verts_uvs = original_verts_uvs,
            mesh = current_cow_mesh,
            weights = loss_weights,
            opt_type = optimization_target
        )

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
cameras_list = build_fixed_cameras(12)
save_render(renderer, final_cow_mesh, cameras_list, output_path+"/final_render") # fixed and not random views
IO().save_mesh(final_cow_mesh, output_path+"/final.obj")