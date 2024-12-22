import torch
from torch.nn import functional as F
from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency

from style_transfer import *

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#method for the second approach
def compute_perceptual_loss(current_imgs, content_imgs, style_imgs, model, style_weight=1e6, content_weight=1):
    
    # Ensure content_imgs and style_imgs are batched tensors
    assert current_imgs.shape[0] == content_imgs.shape[0] == style_imgs.shape[0]

    # Extract features for content and style images
    content_features = get_features(content_imgs, model)['conv4_2']
    style_features = get_features(style_imgs, model)

    # each layer has a shape of 16x64x512x512

    # Calculate style Gram matrices for each image in the batch
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    style_grams.pop('conv4_2')

    # Extract features for current target batch
    current_imgs_features = get_features(current_imgs, model)

    # Calculate content loss
    content_loss = torch.mean((current_imgs_features['conv4_2'] - content_features) ** 2)

    # Calculate style loss
    style_loss = 0
    for layer in style_grams:
        current_feature = current_imgs_features[layer]
        current_gram = gram_matrix(current_feature)
        layer_style_loss = torch.mean((current_gram - style_grams[layer]) ** 2)
        style_loss += layer_style_loss / (current_feature.shape[1] ** 2 * current_feature.shape[2] ** 2)

    # Total loss for this image
    total_loss = content_weight * content_loss + style_weight * style_loss

    return total_loss


# doesn't bring better results if the clipping is done correctly
def rgb_range_loss(mesh):
    texture = mesh.textures.maps_padded()
    loss = torch.sum(torch.relu(texture - 1) + torch.relu(-texture))
    return loss


def compute_first_approach_loss(rendered, masks, target_rendered, verts, target_verts, mesh, weights, opt_type):

    # Compute masked MSE loss for all views in batch
    rendered = rendered * masks  # Shape: [batch_size, C, H, W]
    target_rendered = target_rendered * masks  # Shape: [batch_size, C, H, W]

    if opt_type == 'texture':
        loss = F.mse_loss(rendered, target_rendered) #loss weight ignored (no interest)
        # loss += rgb_range_loss(mesh)
    
    # add mesh optimization loss terms
    elif opt_type == 'mesh':
        loss = weights['main_loss_weight'] * F.mse_loss(rendered, target_rendered)
        loss += weights['mesh_verts_weight'] * F.mse_loss(verts, target_verts)
        loss += weights['mesh_edge_loss_weight'] * mesh_edge_loss(mesh)
        loss += weights['mesh_laplacian_smoothing_weight'] * mesh_laplacian_smoothing(mesh)
        loss += weights['mesh_normal_consistency_weight'] * mesh_normal_consistency(mesh)
        # loss += rgb_range_loss(mesh)
    
    elif opt_type == 'both':
        loss = weights['main_loss_weight'] * F.mse_loss(rendered, target_rendered)
        loss += weights['mesh_verts_weight'] * F.mse_loss(verts, target_verts)
        loss += weights['mesh_edge_loss_weight'] * mesh_edge_loss(mesh)
        loss += weights['mesh_laplacian_smoothing_weight'] * mesh_laplacian_smoothing(mesh)
        loss += weights['mesh_normal_consistency_weight'] * mesh_normal_consistency(mesh)
        # loss += rgb_range_loss(mesh)
    
    return loss


def compute_second_approach_loss(current, content, style, model, style_weight, content_weight, verts, target_verts, mesh, weights, opt_type):

    if opt_type == 'texture':
        loss = compute_perceptual_loss(current, content, style, model, style_weight=style_weight, content_weight=content_weight)

    elif opt_type=='mesh':
        loss =  weights['main_loss_weight'] * compute_perceptual_loss(current, content, style, model, style_weight=style_weight, content_weight=content_weight)
        loss += weights['mesh_verts_weight'] * F.mse_loss(verts, target_verts) 
        loss += weights['mesh_edge_loss_weight'] * mesh_edge_loss(mesh)
        loss += weights['mesh_laplacian_smoothing_weight'] * mesh_laplacian_smoothing(mesh)
        loss += weights['mesh_normal_consistency_weight'] * mesh_normal_consistency(mesh)

    elif opt_type=='both':
        loss =  weights['main_loss_weight'] * compute_perceptual_loss(current, content, style, model, style_weight=style_weight, content_weight=content_weight)
        loss += weights['mesh_verts_weight'] * F.mse_loss(verts, target_verts)
        loss += weights['mesh_edge_loss_weight'] * mesh_edge_loss(mesh)
        loss += weights['mesh_laplacian_smoothing_weight'] * mesh_laplacian_smoothing(mesh)
        loss += weights['mesh_normal_consistency_weight'] * mesh_normal_consistency(mesh)
    
    return loss