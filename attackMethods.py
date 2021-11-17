import torch
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np


def lossObjectness(origLabels, predictedLabels): #as in InvisibleCloak
    score = 0
    for i in range(len(predictedLabels["labels"])):
        if predictedLabels["labels"][i] == 1:
            score =+ max(predictedLabels["scores"][i] + 1, 0)*max(predictedLabels["scores"][i] + 1, 0)
    return score

def TV(patch): # TV - total variation penalthy (smooth for patch)
    K = torch.Tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    K = K.view(1, 1, 3, 3).repeat(1, 3, 1, 1)
    output = F.conv2d(torch.unsqueeze(patch, dim = 0),K, padding=1)
    return output.sum()


def generatePatch():
    return torch.rand(3, 200, 200)

def setPatch(image, patch, box, ratio, device):
    #print("image", image.shape)
    #print("patch size", patch.shape)
    if int(box[2] * ratio) == 0 or int(box[3] * ratio) == 0:
        return image

    resizedPatch = T.Resize(size=(int(box[2] * ratio), int(box[3] * ratio)))(patch)
    #print("resized", resizedPatch.shape)
    #patchedImage = image.clone()
    #onesMask = torch.ones_like(patch)
    paddedPatch = T.Pad(padding=(int(box[0] + box[2]/2), int(box[1] + box[3]/2), image.shape[2] - int(box[0] + box[2]/2) - resizedPatch.shape[2], image.shape[1] - int(box[1] + box[3]/2) - resizedPatch.shape[1]))(resizedPatch)
    
    #print("bb", box)
    #print("paddedSize", paddedPatch.shape)
    
    patchMask = T.Pad(padding=(int(box[0] + box[2]/2), int(box[1] + box[3]/2), image.shape[2] - int(box[0] + box[2]/2) - resizedPatch.shape[2], image.shape[1] - int(box[1] + box[3]/2) - resizedPatch.shape[1]))(torch.ones(size=(3,int(box[2]*ratio),int(box[3]*ratio))))

    #return (torch.ones_like(image).cuda() - patchMask.cuda()) * image + paddedPatch.cuda()
    return (torch.ones_like(image).to(device) - patchMask.to(device)) * image + paddedPatch.to(device)
    
    #return ( patchMask[0])
    #resized_imgs = [T.Resize(size=size)(orig_img) for size in (30, 50, 100, orig_img.size)]

    
    
    