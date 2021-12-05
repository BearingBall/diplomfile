import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2

import torch
import torch.nn as nn
import torch.optim as optim
import random

import torchvision
import torchvision.utils
from torchvision.models import detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import dataset as data
import utils as utils
import pickle
import attackMethods as am



def main(folderImages, folderLabels):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    dataset = data.AdversarialDataset((640,640), folderImages=folderImages, folderLabels=folderLabels) # todo: use resize to pull picture in batch
    loss = am.lossObjectness
    batch_size = 6
    train_loader  = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda:0")

    model.eval()
    model = model.float().to(device)

    patch = am.generatePatch()
    patch = patch.to(device)
    patch.requires_grad = True

    TVCoeff = 0.00000001
    GradRate = 0.03

    epoches = 1

    print('Okey, lets go')

    for epoch in range(epoches):
        imageCounter = 0
        for image, label in train_loader:
            imageCounter += batch_size
        
            #cv2.imshow("patch", cv2.cvtColor(cv2.resize(data.TenToIm(patch.clone().detach().to(torch.device("cpu"))/255), (300,300)), cv2.COLOR_RGB2BGR))

            attackedImage = []

            torch.cuda.empty_cache()

            for im in image:
                attackedImage.append(data.ImToTen(im).to(device))

            for attackedIm in attackedImage:
                attackedIm.requires_grad = True

            clearPredict = model(attackedImage)

            for i in range(len(attackedImage)):
                for l in label[i]:
                    attackedImage[i] = am.setPatch(attackedImage[i], patch, l, 0.2, device) 


            predict = model(attackedImage)

            costs = []

            for i in range(len(clearPredict)):
                cost = loss(clearPredict[i], predict[i])
                if cost == 0:
                    continue
                try:
                    grad = torch.autograd.grad(cost, patch, retain_graph=False, create_graph=False,  allow_unused=True)[0]
                    if grad != None:
                        patch = patch - GradRate*grad.sign()
                except:
                    pass
                costs.append(cost.detach().cpu())
            print("ep:", epoch,"epoch_progress:", imageCounter/len(dataset), "loss:", np.mean(np.asarray(costs)))

            utils.SavePatch(patch.cpu(), "patch")
            cv2.imwrite('patch.png', cv2.cvtColor(patch.cpu().permute(1, 2, 0).detach().numpy(), cv2.COLOR_RGB2BGR))


if __name__=='__main__':
    main(sys.argv[1], sys.argv[2])