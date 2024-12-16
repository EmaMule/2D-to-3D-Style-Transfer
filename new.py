import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
from PIL import Image

# Import style transfer utilities
from style_transfer import load_image, style_transfer, get_vgg, tensor_to_image
from torchvision import transforms

# Import PyTorch3D utilities
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV, FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader, PointLights
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
raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1)
lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

# Create a renderer
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
)

# Load style image and VGG model
style_image = load_image(style_image_path).unsqueeze(0).to(device)
style_image_resized = transforms.Resize((512, 512))(style_image)  # Style image as background
vgg = get_vgg()

# Define angles for viewpoints
angles_x = torch.linspace(0, 270, n_views)  # X-axis rotation
angles_y = torch.linspace(90, 270, n_views - 1)  # Y-axis rotation
angles = [(angle.item(), "X") for angle in angles_x] + [(angle.item(), "Y") for angle in angles_y]

# Initialize texture optimization
current_cow_mesh = content_cow_mesh.clone()
texture_map = current_cow_mesh.textures.maps_padded()
texture_map.requires_grad = True
optimizer = torch.optim.Adam([texture_map], lr=0.01)

# Helper function to blend image with background
def apply_background(rendered_image, mask, background):
    return rendered_image * mask + background * (1 - mask)

# Render and optimize for each viewpoint
for i, (angle, axis) in enumerate(angles):
    # Set up camera rotation
    R = RotateAxisAngle(angle, axis=axis, device=device).get_matrix()[..., :3, :3]
    T = torch.tensor([[0.0, 0.0, 3.0]], device=device)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

    # Render the content image
    rendered_content_out = renderer(meshes_world=content_cow_mesh, cameras=cameras, lights=lights)
    content_image = rendered_content_out[0, ..., :3].permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    object_mask = (content_image > 0).float()  # Binary mask

    if use_background == 2:
        content_image = apply_background(content_image, object_mask, style_image_resized)

    # Render the current texture
    rendered_current_out = renderer(meshes_world=current_cow_mesh, cameras=cameras, lights=lights)
    current_image = rendered_current_out[0, ..., :3].permute(2, 0, 1).unsqueeze(0)

    if use_background in [1,2]:
        current_image = apply_background(current_image, object_mask, style_image_resized)

    # Perform style transfer
    styled_image = style_transfer(current_image, content_image, style_image, vgg, steps=n_style_transfer_steps,
                                  style_weight=style_weight, content_weight=content_weight)

    # Save the styled image
    styled_image_np = tensor_to_image(styled_image)
    styled_image_np.save(f"styled_image_view_{i}.png")

    # Prepare styled image for MSE optimization
    styled_image_tensor = transforms.ToTensor()(styled_image_np).unsqueeze(0).to(device)

    # Optimize the texture to match the styled image
    for step in range(n_mse_steps):
        optimizer.zero_grad()
        current_cow_mesh.textures = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_map)
        rendered_output = renderer(meshes_world=current_cow_mesh, cameras=cameras, lights=lights)

        rendered_image = rendered_output[0, ..., :3].permute(2, 0, 1).unsqueeze(0)
        object_mask = (rendered_output[0, ..., 3] > 0).float().unsqueeze(0).unsqueeze(0)

        # Compute loss with the mask applied
        masked_rendered = rendered_image * object_mask
        masked_target = styled_image_tensor * object_mask
        loss = torch.nn.functional.mse_loss(masked_rendered, masked_target)

        # Backpropagation
        loss.backward()
        optimizer.step()

        print(f"View {i}, Step {step}, Loss: {loss.item()}")

# Save final optimized images
optimized_cow_mesh = current_cow_mesh
for i, (angle, axis) in enumerate(angles):
    R = RotateAxisAngle(angle, axis=axis, device=device).get_matrix()[..., :3, :3]
    T = torch.tensor([[0.0, 0.0, 3.0]], device=device)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

    # Render optimized mesh
    rendered_output = renderer(meshes_world=optimized_cow_mesh, cameras=cameras, lights=lights)
    image = rendered_output[0, ..., :3].unsqueeze(0)
    image_np = image.squeeze(0).detach().cpu().numpy()
    image_np = np.clip(image_np, 0.0, 1.0)  # Ensure valid range

    # Save as PNG
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
    image_pil.save(f"rendered_optimized_cow_view_{i}.png")
    print(f"Saved image for view {i}")
