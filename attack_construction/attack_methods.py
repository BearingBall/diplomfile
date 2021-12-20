import torch
import torch
import torchvision.transforms as T
import numpy as np
import data.utils as data_utils
import attack_construction.metrics as attack_metric
import random


def generate_random_patch(resolution=(70, 70)):
    return torch.rand(3, resolution[0], resolution[1])


def insert_patch(image, patch, box, ratio, device, random_place=False):
    patch_size = (int(box[3] * ratio), int(box[2] * ratio))

    if patch_size[0] == 0 or patch_size[1] == 0: # cant insert patch with this box parameters
        return image

    resized_patch = T.Resize(size=patch_size)(patch)

    x_shift = int(box[0] + box[2] * (random.uniform(ratio, 1-ratio) if random_place else 0.5) - patch_size[1]/2)
    y_shift = int(box[1] + box[3] * (random.uniform(ratio, 1-ratio) if random_place else 0.5) - patch_size[0]/2)

    padding = (x_shift, y_shift, image.shape[2] - x_shift - patch_size[1], image.shape[1] - y_shift - patch_size[0])
    padded_patch = T.Pad(padding=padding)(resized_patch)
    patch_mask = T.Pad(padding=padding)(torch.ones(size=(3, patch_size[0], patch_size[1]))).to(device)
    ones = torch.ones_like(image).to(device)
    result = (torch.ones_like(image).to(device) - patch_mask.to(device)) * image + padded_patch
    patch_mask.detach()
    ones.detach()
    return result


def training_step(model, patch, augmentations, images, labels, loss, device, grad_rate):
    torch.cuda.empty_cache()
    attacked_images = []

    for image in images:
        attacked_images.append(data_utils.image_to_tensor(image).to(device))

    #for attackedIm in attackedImage:
    #    attackedIm.requires_grad = True

    #with torch.no_grad():
    #    clear_predict = model(attacked_images)

    patch = patch if augmentations is None else augmentations(patch)

    for i in range(len(attacked_images)):
        for label in labels[i]:
            attacked_images[i] = insert_patch(attacked_images[i], patch, label, 0.3, device)

    predict = model(attacked_images)

    costs = []

    for i in range(len(predict)):
        cost = loss(predict[i], patch, device)
        if cost < 0.1:
            continue
        try:
            grad = torch.autograd.grad(cost, patch, retain_graph=False, create_graph=False,  allow_unused=True)[0]
            if grad:
                patch = patch - grad_rate * grad.sign()
        except:
             pass
        costs.append(cost.detach().cpu())

    for attacked_image in attacked_images:
        attacked_image.detach()
    
    return np.mean(np.asarray(costs)), patch


def validate(model, patch, augmentations, val_loader, loss_func, device):
    torch.cuda.empty_cache()
    
    losses_before = []
    losses_after = []
    mAPs = []

    for images, labels in val_loader:
        attacked_images = []

        for image in images:
            attacked_images.append(data_utils.image_to_tensor(image).to(device))

        with torch.no_grad():         
            clear_predict = model(attacked_images)

        augmented_patch = patch if augmentations is not None else augmentations(patch)

        for i in range(len(attacked_images)):
            for label in labels[i]:
                attacked_images[i] = insert_patch(attacked_images[i], augmented_patch, label, 0.3, device) 

        with torch.no_grad():        
            predict = model(attacked_images)

        losses_before = losses_before + [loss_func(clear_predict[i], patch, device).detach().cpu() for i in range(len(clear_predict))]
        losses_after = losses_after + [loss_func(predict[i], patch, device).detach().cpu() for i in range(len(predict))]
        mAPs = mAPs + [attack_metric.mean_average_precision(predict[i], labels[i]) for i in range(len(predict))]

    print(mAPs)
    return np.mean(np.asarray(losses_before)), np.mean(np.asarray(losses_after)), np.mean(np.asarray(mAPs))
