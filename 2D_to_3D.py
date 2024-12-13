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
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
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

# Helper functions for style transfer
def load_image(image_path, max_size=512):
    image = Image.open(image_path).convert('RGB')
    size = max_size if max(image.size) > max_size else max(image.size)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = transform(image)[:3, :, :]
    return image.to(device)

def tensor_to_image(tensor):
    image = tensor.clone().detach().squeeze(0)
    image = image * torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    image = image.clamp(0, 1)
    return transforms.ToPILImage()(image.cpu())

def get_vgg():
    vgg = models.vgg19(pretrained=True).features.to(device)
    for param in vgg.parameters():
        param.requires_grad_(False)
    return vgg

def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',
            '28': 'conv5_1'
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    batch_size, d, h, w = tensor.size()
    tensor = tensor.view(batch_size, d, h * w)
    gram = torch.bmm(tensor, tensor.transpose(1, 2))
    return gram

def style_transfer(content_img, style_img, model, steps=200, style_weight=1e6, content_weight=1):
    content_features = get_features(content_img, model)
    style_features = get_features(style_img, model)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    target = content_img.clone().requires_grad_(True).to(device)
    optimizer = torch.optim.Adam([target], lr=0.003)
    for step in tqdm(range(steps)):
        target_features = get_features(target, model)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        style_loss = 0
        for layer in style_grams:
            target_gram = gram_matrix(target_features[layer])
            style_loss += torch.mean((target_gram - style_grams[layer]) ** 2) / (target_features[layer].shape[1] ** 2)
        total_loss = content_weight * content_loss + style_weight * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"Step {step}, Loss: {total_loss.item()}")
    return target

# Load style image
style_image_path = './imgs/Style_1.jpg'
style_image = load_image(style_image_path).unsqueeze(0)

# Load VGG model
vgg = get_vgg()

# Render the cow from multiple viewpoints and perform style transfer
for i, angle in enumerate(angles):
    axis = rotation_axes[i % len(rotation_axes)]
    R = RotateAxisAngle(angle, axis=axis, device=device)
    T = torch.tensor([[0.0, 0.0, 3.0]], device=device)
    cameras = FoVPerspectiveCameras(R=R.get_matrix(), T=T, device=device)

    # Render the image
    rendered_image = renderer(meshes_world=cow_mesh, cameras=cameras, lights=lights)[0, ..., :3]

    # Convert rendered image to format suitable for style transfer
    content_image = rendered_image.permute(2, 0, 1).unsqueeze(0)
    content_image = (content_image - content_image.min()) / (content_image.max() - content_image.min())  # Normalize

    # Perform style transfer
    styled_image = style_transfer(content_image, style_image, vgg, steps=200)

    # Use styled image as target for texture optimization
    styled_image_np = tensor_to_image(styled_image)
    styled_image_np.save(f"styled_image_view_{i}.png")  # Save styled image

    # Convert styled image back to tensor for texture optimization
    styled_image_tensor = transforms.ToTensor()(styled_image_np).unsqueeze(0).to(device)

    # Optimize texture
    optimizer = torch.optim.Adam([cow_mesh.textures.maps_padded()], lr=0.01)
    for step in range(200):
        optimizer.zero_grad()
        rendered_image = renderer(meshes_world=cow_mesh, cameras=cameras, lights=lights)[0, ..., :3]
        rendered_image = rendered_image.permute(2, 0, 1).unsqueeze(0)
        loss = torch.nn.functional.mse_loss(rendered_image, styled_image_tensor)
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"View {i}, Step {step}, Loss: {loss.item()}")

# Save final optimized texture
final_texture = cow_mesh.textures.maps_padded().cpu().detach()
torch.save(final_texture, "optimized_cow_texture.pt")
