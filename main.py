import torch
import matplotlib.pyplot as plt
import os
import numpy as np

from style_transfer import load_image, style_transfer, get_vgg, tensor_to_image
from torchvision import transforms
from PIL import Image

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRenderer,
    MeshRasterizer, SoftPhongShader, PointLights
)

# Set the device (use GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Path to the cow OBJ file
cow_obj_path = "./objects/cow_mesh/cow.obj"

# Load the cow mesh
verts, faces, aux = load_obj(cow_obj_path)
verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
tex_maps = aux.texture_images

verts_uvs = verts_uvs.to(device)
faces_uvs = faces_uvs.to(device)

# Take the first texture image
texture_image = list(tex_maps.values())[0]
texture_image = texture_image[None, ...].to(device)  # (1, H, W, 3)

# Create a textures object
original_textures = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)

# Create the mesh
content_cow_mesh = Meshes(verts=[verts], faces=[faces.verts_idx], textures=original_textures).to(device)

# Camera and renderer settings
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])  # Light source at (0, 0, 3)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=FoVPerspectiveCameras(device=device), raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, cameras=FoVPerspectiveCameras(device=device), lights=lights)
)

# Load the style image
style_image_path = "./imgs/Style_1.jpg"
style_image = load_image(style_image_path).unsqueeze(0)

# Load the VGG model for style transfer
vgg = get_vgg()

# Clone the original mesh for texture optimization
current_cow_mesh = content_cow_mesh.clone()
texture_map = current_cow_mesh.textures.maps_padded()
texture_map.requires_grad = True
optimizer = torch.optim.Adam([texture_map], lr=0.01)

# Define camera positions for the six views
camera_positions = [
    torch.tensor([[3.0, 0.0, 0.0]], device=device),  # +X axis (front)
    torch.tensor([[0.0, 3.0, 0.0]], device=device),  # +Y axis (right)
    torch.tensor([[-3.0, 0.0, 0.0]], device=device), # -X axis (back)
    torch.tensor([[0.0, -3.0, 0.0]], device=device), # -Y axis (left)
    torch.tensor([[0.0, 0.0, 3.0]], device=device),  # +Z axis (above)
    torch.tensor([[0.0, 0.0, -3.0]], device=device)  # -Z axis (below)
]

# Render and optimize texture for each view
for i, position in enumerate(camera_positions):
    T = position
    R = torch.eye(3, device=device).unsqueeze(0)  # Identity rotation matrix
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

    # Render the mesh with the original texture
    rendered_content = renderer(meshes_world=content_cow_mesh, cameras=cameras, lights=lights)[0, ..., :3]
    content_image = rendered_content.permute(2, 0, 1).unsqueeze(0)

    # Perform style transfer
    styled_image = style_transfer(content_image, style_image, vgg, steps=500)
    styled_image_np = tensor_to_image(styled_image)
    styled_image_np.save(f"styled_image_view_{i}.png")

    # Convert styled image back to tensor for texture optimization
    styled_image_tensor = transforms.ToTensor()(styled_image_np).unsqueeze(0).to(device)

    # Optimize texture to match the styled image
    for step in range(200):
        optimizer.zero_grad()
        current_cow_mesh.textures = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_map)
        rendered_image = renderer(meshes_world=current_cow_mesh, cameras=cameras, lights=lights)[0, ..., :3]
        rendered_image = rendered_image.permute(2, 0, 1).unsqueeze(0)
        loss = torch.nn.functional.mse_loss(rendered_image, styled_image_tensor)
        loss.backward()
        optimizer.step()
        print(f"View {i}, Step {step}, Loss: {loss.item()}")

# Save the optimized mesh from each viewpoint
optimized_cow_mesh = current_cow_mesh
for i, position in enumerate(camera_positions):
    T = position
    R = torch.eye(3, device=device).unsqueeze(0)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

    # Render the optimized mesh
    image = renderer(meshes_world=optimized_cow_mesh, cameras=cameras, lights=lights)[0, ..., :3].unsqueeze(0)
    image_np = image.squeeze(0).detach().cpu().numpy()
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
    image_pil.save(f"rendered_optimized_cow_view_{i}.png")
    print(f"Saved optimized image for view {i}")