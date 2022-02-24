import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms
import xml.etree.ElementTree as ET
import json
import numpy as np
import cv2 as cv2
import torch
from os import listdir

class vocDataset(data.Dataset):

    def __init__(self, root, resize=None):
        self.root = root
        self.resize_param = resize    
        self.imagesDir = Path(root) / "JPEGImages"
        self.annDir = Path(root) / "Annotations"
        self.list = [os.path.splitext(filename)[0] for filename in os.listdir(self.imagesDir)]
        

    def __len__(self):
        return len(self.list)


    def __getitem__(self, index):
        image = cv2.imread((self.imagesDir / self.list[index]).as_posix() + '.jpg')
        if image is None:
            raise RuntimeError(
                f'Wrong Image path'
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        tree = ET.parse((self.annDir / self.list[index]).as_posix() + '.xml')
        root = tree.getroot()

        max_labels = 10  # count of labels for one image packed in tensors
        labels_count = 0

        labels_package = torch.zeros((max_labels, 4), dtype=float)

        for object in root.iter('object'):
            if object.find("name").text == "person" and labels_count < max_labels:
                bbox = object.find("bndbox")
                labels_package[labels_count] = torch.tensor([   
                                                            float(bbox.find("xmin").text),
                                                            float(bbox.find("ymin").text),
                                                            float(bbox.find("xmax").text),
                                                            float(bbox.find("ymax").text)
                                                            ])
                labels_count += 1

        scale_factor = None

        if self.resize_param is not None:
            x_ratio = self.resize_param[0]/image.shape[1]
            y_ratio = self.resize_param[1]/image.shape[0]

            scale_factor = torch.tensor([x_ratio, y_ratio])

            for label in labels_package:
                label[0] = int(label[0] * x_ratio)
                label[1] = int(label[1] * y_ratio)
                label[2] = min(int(label[2] * x_ratio), self.resize_param[0] - label[0])
                label[3] = min(int(label[3] * y_ratio), self.resize_param[1] - label[1])

            image = cv2.resize(image, dsize=self.resize_param)

        image: np.ndarray = np.transpose(image, axes=(2, 0, 1)) / 255.
        return torch.tensor(image.astype(np.float32)), labels_package, torch.tensor(index), scale_factor