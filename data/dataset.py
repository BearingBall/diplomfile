import json
import numpy as np
import cv2 as cv2
import torch
from torch.utils.data import Dataset


class AdversarialDataset(Dataset):
    def __init__(
            self,
            resize=None,
            folder_images='../../train2017/train2017',
            labels_file='../../annotations_trainval2017/annotations/instances_train2017.json',
    ):
        self.folder_images = folder_images
        self.labels_file = labels_file
        self.resize_param = resize

        with open(labels_file) as json_file:
            data = json.load(json_file)

        self.images = data['images']
        self.annotations = data['annotations']

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = cv2.imread(self.folder_images + "/" + self.images[index]["file_name"])
        if image is None:
            raise RuntimeError(
                f'Wrong Image path or smth: {self.folder_images + "/" + self.images[index]["file_name"]}'
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        indexes = [i for i in range(len(self.annotations)) if self.annotations[i]["image_id"] == self.images[index]["id"]]

        labels = np.asarray(self.annotations)[indexes]

        max_labels = 10  # count of labels for one image packed in tensors
        labels_count = 0

        labels_package = torch.zeros((max_labels, 4), dtype=float)

        for label in labels:
            if label["category_id"] == 1 and labels_count < max_labels:
                labels_package[labels_count] = torch.tensor(label["bbox"])
                labels_count += 1

        scale_factor = None

        if self.resize_param is not None:
            x_ratio = self.resize_param[0]/image.shape[1]
            y_ratio = self.resize_param[1]/image.shape[0]

            scale_factor = torch.tensor((x_ratio, y_ratio))

            for label in labels_package:
                label[0] = int(label[0] * x_ratio)
                label[1] = int(label[1] * y_ratio)
                label[2] = min(int(label[2] * x_ratio), self.resize_param[0] - label[0])
                label[3] = min(int(label[3] * y_ratio), self.resize_param[1] - label[1])

            image = cv2.resize(image, dsize=self.resize_param)

        image: np.ndarray = np.transpose(image, axes=(2, 0, 1)) / 255.
        return torch.tensor(image.astype(np.float32)), labels_package, self.images[index]["id"], image_size

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