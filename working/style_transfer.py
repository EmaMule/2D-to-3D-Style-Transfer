import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def style_transfer(initial_optimized_imgs, content_imgs, style_imgs, model, steps=2000, style_weight=1e6, content_weight=1, lr=0.003):

    # Ensure content_imgs and style_imgs are batched tensors
    assert initial_optimized_imgs.shape[0] == content_imgs.shape[0] == style_imgs.shape[0]

    # Extract features for content and style images
    content_features = get_features(content_imgs, model)['conv4_2']
    style_features = get_features(style_imgs, model)

    # each layer has a shape of 16x64x512x512

    # Calculate style Gram matrices for each image in the batch
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    style_grams.pop('conv4_2')

    # Initialize target images  --> the ones to optimize
    optimized_imgs = initial_optimized_imgs.clone().detach().requires_grad_(True).to(device)

    # Define optimizer
    optimizer = optim.Adam([optimized_imgs], lr=lr)

    for step in tqdm(range(steps)):

        # Extract features for current target batch
        optimized_imgs_features = get_features(optimized_imgs, model)

        # Calculate content loss
        content_loss = torch.mean((optimized_imgs_features['conv4_2'] - content_features) ** 2)

        # Calculate style loss
        style_loss = 0
        for layer in style_grams:
            optimized_feature = optimized_imgs_features[layer]
            optimized_gram = gram_matrix(optimized_feature)
            layer_style_loss = torch.mean((optimized_gram - style_grams[layer]) ** 2)
            style_loss += layer_style_loss / (optimized_feature.shape[1] ** 2 * optimized_feature.shape[2] ** 2)

        # Total loss for this image
        total_loss = content_weight * content_loss + style_weight * style_loss

        # Backpropagation and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return optimized_imgs


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