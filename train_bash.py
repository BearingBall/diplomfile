import sys
sys.path.append('../')

import numpy as np
import cv2 as cv2

import torch
import torch.hub

import torchvision
import torchvision.utils

from data import dataset as data
import data.utils as data_utils
import attack_construction.attack_methods as attack_methods
import attack_construction.metrics as metrics
import attack_construction.utils as attack_utils
import pickle

print(torch.__version__)
print(torch.cuda_version)
print(torchvision.__version__)



def main(folderImages, folderLabels):
    device = torch.device("cpu")
    #device = torch.device("cuda:0")

    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    model.eval()
    model = model.float().to(device)

    for param in model.parameters():
        param.requires_grad = False 

    dataset = data.AdversarialDataset((640,640), folderImages, folderLabels) # todo: use resize to pull picture in batch

    batch_size = 1

    train_loader  = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    patch = attack_methods.generate_random_patch()
    patch = patch.to(device)
    patch.requires_grad = True

    augmentations = torchvision.transforms.Compose([
        torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.2, hue=0.05),
        torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 0.3)),
        torchvision.transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
        torchvision.transforms.RandomRotation(degrees=(-30, 30)),])

    variation_coef = pow(10, -11)
    grad_rate = 0.03
    epoches = 1

    def loss_function(clear_predict, predict, patch, device):
        return metrics.general_objectness(predict) + variation_coef * metrics.total_variation(patch)

    for epoch in range(epoches):
        image_counter = 0

        for images, labels in train_loader:
            image_counter += batch_size
            loss, patch = attack_methods.training_step(model, patch, augmentations, images, labels, loss_function, device, grad_rate)
            print("ep:", epoch,"epoch_progress:", image_counter/len(dataset), "loss:", loss)

    attack_utils.save_tensor(patch.cpu(), "patch - new")


if __name__=='__main__':
    main(sys.argv[1], sys.argv[2])