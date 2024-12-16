import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np

# Import functions for performing 2D Style Transfer
from style_transfer import load_image, style_transfer, get_vgg, tensor_to_image

# Import transforms
from torchvision import transforms

from PIL import Image

# Import loading of mesh / obj
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV

# To rotate around the object
from pytorch3d.transforms import RotateAxisAngle

from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRenderer,
    MeshRasterizer, SoftPhongShader, PointLights
)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_views", default=4,type=int, help="Number of views considered by the renderer")
parser.add_argument("--n_mse_steps", default=100, type=int, help="Number steps mse")
parser.add_argument("--n_style_transfer_steps", default=3000, type=int, help="Number steps style transfer")
parser.add_argument("--use_background", default=0, type=int, help="Specify background modality: if 0, no background, if 1 background on the rendered image, 2 background on both")
parser.add_argument("--obj_path", default="./objects/cow_mesh/cow.obj", type=str, help="Specify the path to the object")
parser.add_argument("--style_path", default="./imgs/Style_1.jpg", type=str, help="Specify the path to the style")
parser.add_argument("--style_weight", default=1e6, type=int, help="Weight of the style")
parser.add_argument("--content_weight", default=1, type=int, help="Weight of the content")
args = parser.parse_args()

#parsing args
cow_obj_path = args.obj_path
style_image_path = args.style_path
n_views = args.n_views
n_mse_steps = args.n_mse_steps
n_style_transfer_steps = args.n_style_transfer_steps
use_background = args.use_background

content_weight = args.content_weight
style_weight = args.style_weight

# Helper function to blend image with background
def apply_background(rendered_image, mask, background):
    return rendered_image * mask + background * (1 - mask)

# Set the device (use GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the cow mesh
verts, faces, aux = load_obj(cow_obj_path)

verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
tex_maps = aux.texture_images

verts_uvs = verts_uvs.to(device)
faces_uvs = faces_uvs.to(device)

# tex_maps is a dictionary of {material name: texture image}.
# Take the first image:
texture_image = list(tex_maps.values())[0]
texture_image = texture_image[None, ...].to(device)  # (1, H, W, 3)

# Create a textures object
original_textures = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)

# Initialise the mesh with textures
content_cow_mesh = Meshes(verts=[verts], faces=[faces.verts_idx], textures=original_textures)
content_cow_mesh = content_cow_mesh.to(device)  # Move the mesh to the right device

# Camera settings
cameras = FoVPerspectiveCameras(device=device)

# Rasterization settings
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# Lights settings
lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]]) #might be useful to use at least 2

# Create a renderer
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
)

style_image = load_image(style_image_path).unsqueeze(0)  # If we end up working in batches, we need to stack them

# Load VGG model
vgg = get_vgg()

# Define multiple viewpoints for both X-axis and Y-axis rotations
angles_x = torch.linspace(0, 270, n_views)  # Angles in degrees for X-axis rotation (no repetition)
angles_y = torch.linspace(90, 270, n_views-1)  # Angles in degrees for Y-axis rotation (no repetition)

# Combine the angles and axes into a single list
angles = [(angle.item(), "X") for angle in angles_x] + [(angle.item(), "Y") for angle in angles_y]

# Current texture
current_cow_mesh = content_cow_mesh.clone()  # Start as the original

# Optimize texture
texture_map = current_cow_mesh.textures.maps_padded()
texture_map.requires_grad = True  # Ensure the texture map has requires_grad=True

optimizer = torch.optim.Adam([texture_map], lr=0.01)

# Render the cow from multiple viewpoints and perform style transfer
for i, (angle, axis) in enumerate(angles):
    R = RotateAxisAngle(angle, axis=axis, device=device).get_matrix()[..., :3, :3]  # Extract 3x3 rotation matrix
    T = torch.tensor([[0.0, 0.0, 3.0]], device=device)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

    # Render the content image with the original texture
    rendered_content_out = renderer(meshes_world=content_cow_mesh, cameras=cameras, lights=lights)

    # Get the RGB image and the alpha mask (indicating object visibility)
    content_image = rendered_content_out[0, ..., :3]  # Extract RGB channels
    object_mask = (content_image > 0).float()  # Create binary mask from alpha channel (1 for object pixels, 0 for background)

    # Convert rendered images to suitable format
    content_image = content_image.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    rendered_current_out = renderer(meshes_world=current_cow_mesh, cameras=cameras, lights=lights)
    current_image = rendered_current_out[0, ..., :3]

    current_image = current_image.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)


    # Perform style transfer on the masked cow image --> errore
    styled_image = style_transfer(current_image, content_image, style_image, vgg, steps=n_style_transfer_steps, style_weight=style_weight, content_weight=content_weight)

    # Use styled image as target for texture optimization
    styled_image_np = tensor_to_image(styled_image)
    styled_image_np.save(f"styled_image_view_{i}.png")  # Save styled image

    # Convert styled image back to tensor for texture optimization
    styled_image_tensor = transforms.ToTensor()(styled_image_np).unsqueeze(0).to(device)

    for step in range(n_mse_steps):
        optimizer.zero_grad()

        # Update current mesh texture
        current_cow_mesh.textures = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_map)
        rendered_output = renderer(meshes_world=current_cow_mesh, cameras=cameras, lights=lights)

        rendered_image = rendered_output[0, ..., :3].permute(2, 0, 1).unsqueeze(0)  # RGB channels
        object_mask = (rendered_output[0, ..., 3] > 0).float().unsqueeze(0).unsqueeze(0)  # Binary mask, match dimensions

        # Compute masked loss
        masked_rendered = rendered_image * object_mask  # Apply mask to rendered image
        masked_target = styled_image_tensor * object_mask  # Apply mask to target image

        loss = torch.nn.functional.mse_loss(masked_rendered, masked_target)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        print(f"View {i}, Step {step}, Loss: {loss.item()}")

optimized_cow_mesh = current_cow_mesh  # Final

# Loop over each viewpoint and render the image
for i, (angle, axis) in enumerate(angles):
    R = RotateAxisAngle(angle, axis=axis, device=device).get_matrix()[..., :3, :3]  # Get the 3x3 rotation matrix
    T = torch.tensor([[0.0, 0.0, 3.0]], device=device)  # Translation vector (camera position)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

    # Render the optimized cow mesh from the current viewpoint
    rendered_output = renderer(meshes_world=optimized_cow_mesh, cameras=cameras, lights=lights)
    image = rendered_output[0, ..., :3].unsqueeze(0)

    # Convert the image from tensor to NumPy array
    image_np = image.squeeze(0).detach().cpu().numpy()  # Convert to NumPy array on CPU

    # Clip the values to the range [0, 1]
    image_np = np.clip(image_np, 0.0, 1.0)  # Ensures valid range for saving

    # Convert the image to PIL format
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))  # Convert to uint8 format for saving

    # Save the image for the current viewpoint
    image_pil.save(f"rendered_optimized_cow_view_{i}.png")  # Save the image with a unique name
    print(f"Saved image for view {i}")


