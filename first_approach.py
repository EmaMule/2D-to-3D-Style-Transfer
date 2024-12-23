import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
import math
import argparse

# Import style transfer utilities
from style_transfer import *
from utils import *
from losses import *

# Import PyTorch3D utilities
from pytorch3d.io import load_obj, IO
from pytorch3d.renderer import FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader, AmbientLights

# Set device (use GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--n_views", default=6, type=int, help="Number of views considered by the renderer")
parser.add_argument("--n_mse_steps", default=100, type=int, help="Number of steps for MSE optimization")
parser.add_argument("--n_style_transfer_steps", default=3000, type=int, help="Number of steps for style transfer")
parser.add_argument("--obj_path", default="./objects/cow_mesh/cow.obj", type=str, help="Path to the object")
parser.add_argument("--style_path", default="./imgs/Style_1.jpg", type=str, help="Path to the style image")
parser.add_argument("--style_weight", default=1e6, type=float, help="Weight of the style loss")
parser.add_argument("--content_weight", default=1.0, type=float, help="Weight of the content loss")
parser.add_argument("--resize_texture", default=True, type=bool, help="Whether to resize the texture to the same size of the images")
parser.add_argument("--size", default=768, type=int, help="Dimension of the images")
parser.add_argument("--output_path", default="/content/output_first", type=str, help="Output folder path")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
parser.add_argument("--style_transfer_init", default='content', type=str, choices=['noise', 'current', 'content'], help="Initialization for the 2D Style Transfer")
parser.add_argument("--content_background", default='white', type=str, choices=['noise', 'style', 'white'], help="Type of background for the content image")
parser.add_argument("--current_background", default='white', type=str, choices=['noise', 'style', 'white'], help="Type of background for the current image")
parser.add_argument("--style_transfer_lr", default=0.01, type=float, help="Style Transfer Learning Rate")
parser.add_argument("--mse_lr", default=0.01, type=float, help="2D to 3D Learning Rate")
parser.add_argument("--randomize_views", type=bool, default=True, help="Whether or not to randomize views") #if it is of interest we might do it
parser.add_argument("--optimization_target", type=str, choices=['texture', 'mesh', 'both'], default="texture", help="Decide what to optimize")
parser.add_argument("--main_loss_weight", type=float, default=3.0, help="Weight of the main computed loss (i.e., mse)")
parser.add_argument("--mesh_edge_loss_weight", type=float, default=1.0, help="Weight of edge loss (enforces admissible weights for the edges)")
parser.add_argument("--mesh_laplacian_smoothing_weight", type=float, default=1.0, help="Weight of smoothing (smooth surface)")
parser.add_argument("--mesh_normal_consistency_weight", type=float, default=1.0, help="Weight of normal consistency")
parser.add_argument("--mesh_verts_weight", type=float, default=1.0, help="Mesh verts (uvs and not uvs) regularization weight")

args = parser.parse_args()

# Parse arguments
obj_path = args.obj_path
style_image_path = args.style_path
n_views = args.n_views
n_mse_steps = args.n_mse_steps
n_style_transfer_steps = args.n_style_transfer_steps
content_weight = args.content_weight
style_weight = args.style_weight
resize_texture = args.resize_texture
size = args.size
output_path = args.output_path
batch_size = args.batch_size
style_transfer_init = args.style_transfer_init
content_background = args.content_background
current_background = args.current_background
mse_lr = args.mse_lr
style_transfer_lr = args.style_transfer_lr
randomize_views=args.randomize_views
optimization_target = args.optimization_target

loss_weights = {
    'mesh_edge_loss_weight': args.mesh_edge_loss_weight,
    'mesh_laplacian_smoothing_weight': args.mesh_laplacian_smoothing_weight,
    'mesh_normal_consistency_weight': args.mesh_normal_consistency_weight,
    'mesh_verts_weight': args.mesh_verts_weight,
    'main_loss_weight': args.main_loss_weight,
}

# Create output folder
os.makedirs(output_path, exist_ok=True)
os.makedirs(output_path+"/2d_style_transfer", exist_ok=True)

# Loading mesh
print("Loading mesh...")
original_verts, original_faces, aux = load_obj(obj_path)
original_verts = original_verts.to(device)
original_verts_uvs = aux.verts_uvs[None, ...].to(device)  # (1, V, 2)
original_faces_uvs = original_faces.textures_idx[None, ...].to(device)  # (1, F, 3)
original_faces = original_faces.verts_idx.to(device) #notice I'm overwriting the variable (it is not used in any case)
texture_image = list(aux.texture_images.values())[0][None, ...].to(device)  # (1, H, W, 3)

if resize_texture:
  texture_image = texture_image.permute(0, 3, 1, 2)  # Shape: (1, 3, H, W)
  # Resize the texture
  texture_image = F.interpolate(
      texture_image, 
      size=size, 
      mode='bilinear', 
      align_corners=False
  )
  # Permute back to NHWC format
  texture_image = texture_image.permute(0, 2, 3, 1)  # Shape: (1, H, W, 3)

# Initialize content textures and mesh
content_mesh = build_mesh(original_verts_uvs, original_faces_uvs, texture_image, original_verts, original_faces)

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
print("Loading model...")
vgg = get_vgg()

# Build cameras
print("Building cameras...")
if randomize_views:
    cameras_list = build_random_cameras(n_views)
else:
    cameras_list = build_fixed_cameras(n_views)

# RIVEDERE CON MATTEO: SI PUO FARE DI MEGLIO?
# initialize optimization based on the target (it returns the mesh to optimize etc.)
out = setup_optimizations(optimization_target, content_mesh, mse_lr)

#retrieve outputs (done like this for clarity)
current_mesh = out['optimizable_mesh']
optimizer = out['optimizer']
texture_map = out['texture_map']
verts = out['verts']
faces = out['faces']
verts_uvs = out['verts_uvs']
faces_uvs = out['faces_uvs']

# SHOULD COMPUTE ALL 2D TRANSFORM FIRST AND THEN LEARN WITH BATCHES FOR A FEW ITERATIONS?

# Logging
with open(output_path + '/log.txt', 'w') as file:
    file.write('Logger:\n')

print("Starting optimization...")
for i in range(math.ceil(n_views / batch_size)):

    print(f"\nBatch {i}")

    batch_start = i*batch_size
    batch_end = min((i+1)*batch_size, n_views)
    current_batch_size = batch_end - batch_start

    # Sample cameras (shuffling is done in the angles, they can be taken in order)
    batch_indexes = list(range(batch_start, batch_end))
    batch_cameras = [cameras_list[idx] for idx in batch_indexes]

    # Load style image
    style_tensors = load_as_tensor(style_image_path, size=size).repeat(current_batch_size, 1, 1, 1).to(device)

    # Render content images for all views
    content_tensors, content_masks = render_meshes(renderer, content_mesh, batch_cameras)
    content_tensors = apply_background(content_tensors, content_masks, background_type=content_background, background=style_tensors)

    # Initialize 2d style trasfer tensors
    if style_transfer_init == 'noise':
        applied_style_tensors = torch.rand(content_tensors.shape, device=device)
    elif style_transfer_init == 'content':
        applied_style_tensors = content_tensors
    elif style_transfer_init == 'current':
        current_mesh = build_mesh(verts_uvs, faces_uvs, texture_map, verts, faces)
        current_tensors, current_masks = render_meshes(renderer, current_mesh, batch_cameras)
        current_tensors = apply_background(current_tensors, current_masks, background_type=current_background, background=style_tensors)
        applied_style_tensors = current_tensors

    # Perform batch style transfer
    applied_style_tensors = style_transfer(applied_style_tensors, content_tensors, style_tensors, vgg, steps=n_style_transfer_steps,
                                    style_weight=style_weight, content_weight=content_weight, lr=style_transfer_lr)

    # the produced values may be outside the range (0,1)
    applied_style_tensors = finalize_tensor(applied_style_tensors)

    # Save styled images
    for j, applied_style_tensor in enumerate(applied_style_tensors):
        applied_style_image = tensor_to_image(applied_style_tensor)
        applied_style_image.save(output_path + f"/2d_style_transfer/view_{i*batch_size+j}.png")

    # Optimize the texture map in batches
    loss_value = 0
    for step in tqdm(range(n_mse_steps), desc="Optimizing", postfix=loss_value):
        optimizer.zero_grad()

        # Done because pytorch otherwise cries
        current_mesh = build_mesh(verts_uvs, faces_uvs, texture_map, verts, faces)

        rendered_tensors, object_masks = render_meshes(renderer, current_mesh, batch_cameras)

        loss = compute_first_approach_loss(
            rendered = rendered_tensors,
            masks = object_masks,
            target_rendered = applied_style_tensors, 
            verts = verts, 
            target_verts = original_verts, 
            mesh = current_mesh, 
            weights = loss_weights, 
            opt_type = optimization_target
        )

        # Backpropagation
        loss.backward()
        optimizer.step()
        loss_value = loss.item()

        # Logging
        with open(output_path + '/log.txt', 'a') as file:
            file.write(f'Batch {i}, Step {step}, Loss {loss_value}\n')

# Ensure texture values are in the correct range
final_mesh = finalize_mesh(current_mesh)

# Save final optimized images
cameras_list = build_fixed_cameras(12)
save_render(renderer, final_mesh, cameras_list, output_path+"/final_render")
IO().save_mesh(final_mesh, output_path+"/final.obj")