import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

#import functions for performing 2D Style Transfer
from style_transfer import load_image, style_transfer, get_vgg, tensor_to_image

#import transforms
from torchvision import transforms

#import loading of mesh / obj
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV

#to rotate around the object
from pytorch3d.transforms import RotateAxisAngle

from pytorch3d.vis.plotly_vis import plot_scene


from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRenderer,
    MeshRasterizer, SoftPhongShader, PointLights, TexturesUV
)
import matplotlib.pyplot as plt

# Set the device (use GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Path to the cow OBJ file
cow_obj_path = "./objects/cow_mesh/cow.obj"

# Load the cow mesh
verts, faces, aux = load_obj(cow_obj_path)

verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
tex_maps = aux.texture_images

# tex_maps is a dictionary of {material name: texture image}.
# Take the first image:
texture_image = list(tex_maps.values())[0]
texture_image = texture_image[None, ...]  # (1, H, W, 3)

# Create a textures object
original_textures = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)

# Initialise the mesh with textures
content_cow_mesh = Meshes(verts=[verts], faces=[faces.verts_idx], textures=original_textures) #notice --> has the texture

content_cow_mesh = content_cow_mesh.to(device) #move the mesh to the right device

# Camera settings
cameras = FoVPerspectiveCameras(device=device)

# Rasterization settings
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

#Remember: the origin of the system is the center of the object itself!

# Lights settings
lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]]) #sets the light source at 3 meters on the z axis

# Create a renderer
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
)

#take the style_image
style_image_path = "./imgs/Style_2.jpg"
style_image = load_image(style_image_path).unsqueeze(0)

# Load VGG model
vgg = get_vgg()

# Define multiple viewpoints
num_views = 20 # Number of viewpoints
angles = torch.linspace(0, 360, num_views)  # Angles in degrees
rotation_axes = ["X", "Y", "Z"]  # Rotate around different axes


#current texture
current_cow_mesh = content_cow_mesh.clone() #start as the original

# Optimize texture
texture_map = current_cow_mesh.textures.maps_padded()
texture_map.requires_grad = True  # Ensure the texture map has requires_grad=True

optimizer = torch.optim.Adam([texture_map], lr=0.01)
# Render the cow from multiple viewpoints and perform style transfer
for i, angle in enumerate(angles):
    axis = rotation_axes[i % len(rotation_axes)]
    R = RotateAxisAngle(angle, axis=axis, device=device).get_matrix()[..., :3, :3]  # Extract 3x3 rotation matrix
    T = torch.tensor([[0.0, 0.0, 3.0]], device=device)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

    # Render the content image with the original textur
    rendered_content = renderer(meshes_world=content_cow_mesh, cameras=cameras, lights=lights)[0, ..., :3] #render the current cow

    # Convert rendered images to suitable format
    content_image = rendered_content.permute(2, 0, 1).unsqueeze(0)
    # content_image = (content_image - content_image.min()) / (content_image.max() - content_image.min())  # Normalize

    # Perform style transfer on the masked cow image
    styled_image = style_transfer(content_image, style_image, vgg, steps=1)

    # Use styled image as target for texture optimization
    styled_image_np = tensor_to_image(styled_image)
    styled_image_np.save(f"styled_image_view_{i}.png")  # Save styled image

    # Convert styled image back to tensor for texture optimization
    styled_image_tensor = transforms.ToTensor()(styled_image_np).unsqueeze(0).to(device)

    for step in range(2):
        optimizer.zero_grad()
        
        current_cow_mesh.textures = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_map)
        rendered_image = renderer(meshes_world=current_cow_mesh, cameras=cameras, lights=lights)[0, ..., :3]
        rendered_image = rendered_image.permute(2, 0, 1).unsqueeze(0)
        loss = torch.nn.functional.mse_loss(rendered_image, styled_image_tensor)
        loss.backward()
        optimizer.step()
        print(f"View {i}, Step {step}, Loss: {loss.item()}")

optimized_cow_mesh = current_cow_mesh #final

fig = plot_scene({
    "subplot1": {
        "cow_mesh": optimized_cow_mesh
    }
})
fig.show()