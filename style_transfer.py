import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess the images
def load_image(image_path, max_size=512):
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

    image = transform(image)[:3, :, :]
    return image.to(device)

# Convert tensor to image for display
def tensor_to_image(tensor):
    image = tensor.clone().detach()
    image = image.squeeze(0)  # Remove batch dimension
    image = image * torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406], device = device).view(3, 1, 1)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image.cpu())
    return image

# Load the pre-trained VGG19 model
def get_vgg():
    vgg = models.vgg19(pretrained=True).features.to(device)
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
    batch_size, d, h, w = tensor.size()
    tensor = tensor.view(batch_size, d, h * w)  # Flatten the spatial dimensions.
    gram = torch.bmm(tensor, tensor.transpose(1, 2))  # Batch matrix multiplication.
    return gram

def style_transfer(optim_imgs, content_imgs, style_imgs, model, steps=2000, style_weight=1e6, content_weight=1):

    # Ensure content_imgs and style_imgs are batched tensors
    assert optim_imgs.shape[0] == content_imgs.shape[0] == style_imgs.shape[0], "Batch sizes of content and style images must match."

    # Extract features for content and style images
    content_features = get_features(content_imgs, model)
    style_features = get_features(style_imgs, model)

    #each layer has a shape of 16x64x512x512

    # Calculate style Gram matrices for each image in the batch
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # Initialize target images  --> the ones to optimize
    targets = optim_imgs.clone().detach().requires_grad_(True).to(device)


    # Define optimizer
    optimizer = optim.Adam([targets], lr=0.003)

    for step in tqdm(range(steps)):

        # Extract features for current target batch
        targets_features = get_features(targets, model)

        # Calculate content loss
        content_loss = torch.mean((targets_features['conv4_2'] - content_features['conv4_2']) ** 2)

        # Calculate style loss
        style_loss = 0
        for layer in style_grams:
            target_feature = targets_features[layer]
            target_gram = gram_matrix(target_feature)
            layer_style_loss = torch.mean((target_gram - style_grams[layer]) ** 2)
            style_loss += layer_style_loss / (target_feature.shape[1] ** 2 * target_feature.shape[2] ** 2)

        # Total loss for this image
        total_loss = content_weight * content_loss + style_weight * style_loss

        # Backpropagation and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return targets


# # Load content and style images
# content = torch.stack([load_image('./imgs/Content.jpg') for _ in range(3)]).to(device)
# style = torch.stack([load_image('./imgs/Style_1.jpg'), load_image('./imgs/Style_2.jpg'), load_image('./imgs/Style_3.jpg')]).to(device)
# # Load VGG model
# vgg = get_vgg()

# # Perform style transfer
# output = style_transfer(content, style, vgg, steps = 8000, style_weight=10e4, content_weight=10)

# # Display the result
# # Save and display each result in the batch
# for i, img in enumerate(output):
#     result = tensor_to_image(img)
#     plt.figure()
#     plt.imshow(result)
#     plt.axis('off')
#     plt.show()

#     # Save the output
#     result.save(f'output_{i}.jpg')
