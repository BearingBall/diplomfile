import os
import json
import numpy as np
import cv2 as cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

def ImToTen(image):
    return (torch.tensor(np.swapaxes(np.swapaxes(image, 0, 2),1,2)).float() / 255)

def TenToIm(tensor):
    return tensor.permute(1, 2, 0).detach().numpy()*255


class AdversarialDataset(Dataset):
    def __init__(self, resize = None, folderImages = "../train2017/train2017", folderLabels = "../annotations_trainval2017/annotations/instances_train2017.json"):
        self.folderImagesName = folderImages
        self.folderLabelsName = folderLabels
        self.resizeParam = resize

        with open(folderLabels) as json_file:
            data = json.load(json_file)

        self.images = data['images']
        self.annotations = data['annotations']

        #self.fileNames = os.listdir(folder)

    def __len__(self):
        return len(self.images)
        #return 2000
    
    def __getitem__(self, index):    
        if (index > len(self.images)):
            return None

        image = cv2.imread(self.folderImagesName+"/"+self.images[index]["file_name"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        indexes = [i for i in range(len(self.annotations)) if self.annotations[i]["image_id"] == self.images[index]["id"]]

        labels = np.asarray(self.annotations)[indexes]

        tenLabels = []

        labelsNumber = 0
        labelsSize = 5

        for l in labels:
            if l["category_id"] == 1 and labelsNumber < labelsSize:
                labelsNumber+=1
                tenLabels.append(torch.tensor(l["bbox"]))

        if (self.resizeParam != None):
            image = cv2.resize(image, self.resizeParam)
            x_ratio = self.resizeParam[0]/image.shape[0]
            y_ratio = self.resizeParam[1]/image.shape[1]
            for l in tenLabels:
                l[0] = int(l[0] * x_ratio)
                l[2] = int(l[2] * x_ratio)
                l[1] = int(l[1] * y_ratio)
                l[3] = int(l[3] * y_ratio)

        for i in range(labelsNumber, labelsSize):
            tenLabels.append(torch.tensor(np.array([0.0,0.0,0.0,0.0])))

        return image, tenLabels

"""
{
    "info": {...},
    "licenses": [],
    "categories": [
        {
            "id": 0,
            "name": "airplane",
            "supercategory": null
        },
        ...
    ],
    "images": [
        {
            "id": 1,
            "file_name": "001631.jpg",
            "height": 612,
            "width": 612,
            "license": null,
            "coco_url": null
        },
        ...
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 9,
            "bbox": [
                92.14,
                220.04,
                519.86,
                61.89000000000001
            ],
            "area": 32174.135400000006,
            "iscrowd": 0
        },
        ...
    ]
}
"""