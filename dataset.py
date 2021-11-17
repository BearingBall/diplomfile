import os
import json
import numpy as np
import cv2 as cv2
from torch.utils.data import Dataset
import torch

def ImToTen(image):
    return (torch.tensor(np.swapaxes(np.swapaxes(image, 0, 2),1,2)).float() / 255)

def TenToIm(tensor):
    return tensor.permute(1, 2, 0).detach().numpy()


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
    
    def __getitem__(self, index):    
        if (index > len(self.images)):
            return None

        image = cv2.imread(self.folderImagesName+"/"+self.images[index]["file_name"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if (self.resizeParam != None):
            image = cv2.resize(image, self.resizeParam)

        indexes = [i for i in range(len(self.annotations)) if self.annotations[i]["image_id"] == self.images[index]["id"]]

        return image, np.asarray(self.annotations)[indexes]

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