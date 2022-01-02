import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import attack_construction.attack_methods as am
import pickle
import torch
from pathlib import Path
import data.utils as data_utils


def show_tensor(picture):
    plt.imshow(data_utils.tensor_to_image(picture)/255)


def save_patch_tensor(tensor, path, epoch=None, step=None, save_mode='both'):
    condition = epoch is None or step is None
    pickle_folder = Path(path) / 'pickle'
    pickle_folder.mkdir(parents=True, exist_ok=True)
    image_folder = Path(path) / 'image'
    image_folder.mkdir(parents=True, exist_ok=True)

    if save_mode == 'pickle' or save_mode == 'both':
        pickle_path = 'patch.pickle' if condition else f'patch_{epoch}_{step}.pickle'
        with open(pickle_folder / pickle_path, 'wb') as f:
            pickle.dump(data_utils.tensor_to_image(tensor), f)

    if save_mode == 'img' or save_mode == 'both':
        img_path = 'patch.png' if epoch is None else f'patch_{epoch}_{step}.png'
        cv2.imwrite((image_folder / img_path).as_posix(), data_utils.tensor_to_image(tensor))


def load_tensor_from_pickle(path_to_pickle):
    with open(path_to_pickle, 'rb') as f:
        return data_utils.image_to_tensor(pickle.load(f))


def load_tensor_from_image(path_to_image):
    image = cv2.cvtColor(cv2.imread(path_to_image), cv2.COLOR_BGR2RGB)
    return data_utils.image_to_tensor(image)


def visualize_labels_predicted(image, labels, threshold):
    image = np.ascontiguousarray(image.copy(), dtype=np.uint8)

    for i in range(len(labels["labels"])):

        if labels["scores"][i] < threshold:
            continue

        if labels["labels"][i] == 1:
            pt1 = (int(labels['boxes'][i][0]), int(labels['boxes'][i][1]))
            pt2 = (int(labels['boxes'][i][2]), int(labels['boxes'][i][3]))
            border_color = (255, 0, 0)
            text_color = (0, 0, 255)
            image = cv2.rectangle(img=image, pt1=pt1, pt2=pt2, color=border_color, thickness=1)
            object_score = str(float(labels["scores"][i]))
            image = cv2.putText(
                img=image,
                text=object_score,
                org=pt1,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=text_color,
                thickness=2,
                lineType=cv2.LINE_AA,
                bottomLeftOrigin=False,
            )
    return image


def visualize_labels_gt(image, labels):
    image = np.ascontiguousarray(image.copy(), dtype=np.uint8)
    for label in labels:
        pt1 = (int(label[0]), int(label[1]))
        pt2 = (int(label[0] + label[2]), int(label[1] + label[3]))
        border_color = (255, 0, 0)
        image = cv2.rectangle(img=image, pt1=pt1, pt2=pt2, color=border_color, thickness=1)
    return image


def get_patch_test(patch, image, label, model, device, threshold=0.0):
    image = data_utils.image_to_tensor(image).to(device)
    attacked_image = image.clone()

    for l in label:
        attacked_image = am.insert_patch(attacked_image, patch, l, 0.3, device) 

    with torch.no_grad():
        prediction = model([image, attacked_image])

    image_before = visualize_labels(data_utils.tensor_to_image(image), prediction[0], threshold)
    image_after = visualize_labels(data_utils.tensor_to_image(attacked_image), prediction[1], threshold)

    return image_before, image_after
