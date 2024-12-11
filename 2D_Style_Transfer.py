import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load and preprocess the images
def load_image(image_path, max_size=400):
    image = Image.open(image_path).convert('RGB')

    # Resize the image
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Add batch dimension
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image

# Convert tensor to image for display
def tensor_to_image(tensor):
    image = tensor.clone().detach()
    image = image.squeeze(0)  # Remove batch dimension
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    return image

# Load the pre-trained VGG19 model
def get_vgg():
    vgg = models.vgg19(pretrained=True).features
    for param in vgg.parameters():
        param.requires_grad_(False)
    return vgg

# Extract features using VGG19
def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # Content layer
            '28': 'conv5_1'
        }

    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Calculate Gram matrix for style representation
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Define style transfer function
def style_transfer(content_img, style_img, model, steps=2000, style_weight=1e6, content_weight=1):
    # Extract features
    content_features = get_features(content_img, model)
    style_features = get_features(style_img, model)
    
    # Calculate style Gram matrices
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # Initialize target image as a copy of the content image
    target = content_img.clone().requires_grad_(True)

    # Define optimizer
    optimizer = optim.Adam([target], lr=0.003)

    for step in tqdm(range(steps)):
        target_features = get_features(target, model)

        # Calculate content loss
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

        # Calculate style loss
        style_loss = 0
        for layer in style_grams:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_style_loss / (target_feature.shape[1] ** 2 * target_feature.shape[2] ** 2)

        # Total loss
        total_loss = content_weight * content_loss + style_weight * style_loss

        # Backpropagation and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}, Total loss: {total_loss.item()}")

    return target

# Load content and style images
content = load_image('./imgs/Content.jpg')
style = load_image('./imgs/Style.jpg')

# Load VGG model
vgg = get_vgg()

# Perform style transfer
output = style_transfer(content, style, vgg)

# Display the result
result = tensor_to_image(output)
plt.imshow(result)
plt.axis('off')
plt.show()

# Save the output
result.save('output.jpg')
