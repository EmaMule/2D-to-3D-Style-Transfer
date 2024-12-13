import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    FoVPerspectiveCameras,
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import RotateAxisAngle
import matplotlib.pyplot as plt
import os

# Set the device (use GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Path to the cow OBJ file
cow_obj_path = "./objects/cow_mesh/cow.obj"

# Load the cow mesh
cow_mesh = load_objs_as_meshes([cow_obj_path], device=device)

# Define a renderer
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# Set up the camera with a perspective projection
camera = FoVPerspectiveCameras(device=device)

# Define a simple point light
lights = PointLights(device=device, location=[[2.0, 2.0, 2.0]])

# Create the renderer with a SoftPhongShader
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera,
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device,
        cameras=camera,
        lights=lights
    )
)

# Define multiple viewpoints
num_views = 20  # Number of viewpoints
angles = torch.linspace(0, 360, num_views)  # Angles in degrees
rotation_axes = ["X", "Y", "Z"]  # Rotate around different axes

# Render the cow from multiple viewpoints
images = []
for i, angle in enumerate(angles):
    axis = rotation_axes[i % len(rotation_axes)]  # Cycle through axes
    R = RotateAxisAngle(angle, axis=axis, device=device)

    # Set the camera distance and apply rotation
    T = torch.tensor([[0.0, 0.0, 3.0]], device=device)  # Camera translation

    cameras = FoVPerspectiveCameras(R=R.get_matrix(), T=T, device=device)

    # Render the image
    image = renderer(meshes_world=cow_mesh, cameras=cameras, lights=lights)
    images.append(image.cpu().numpy())

# Plot the rendered images
fig, axes = plt.subplots(4, 5, figsize=(20, 10))
axes = axes.flatten()
for i, ax in enumerate(axes):
    if i < len(images):
        ax.imshow(images[i][0, ..., :3])  # Only plot RGB channels
        ax.axis("off")
    else:
        ax.axis("off")
plt.show()
