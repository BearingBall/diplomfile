import cv2 as cv2
import matplotlib.pyplot as plt
import attack_construction.attack_methods as am
import pickle
import torch
import data.utils as data_utils


pathes_path = "../patches/"


def show_tensor(picture):
    plt.imshow(data_utils.tensor_to_image(picture)/255)


def save_tensor(tensor, name, external_place = False):
    with open(("" if external_place else pathes_path) + name + '.pickle', 'wb') as f:
        pickle.dump(data_utils.tensor_to_image(tensor), f)
    cv2.imwrite(pathes_path + name + ".png", data_utils.tensor_to_image(tensor))


def load_tensor(name):
    with open(pathes_path + name + '.pickle', 'rb') as f:
        return data_utils.image_to_tensor(pickle.load(f))

def load_tensor_from_image(name):
    image = cv2.cvtColor(cv2.imread(pathes_path + name), cv2.COLOR_BGR2RGB)
    return data_utils.image_to_tensor(image)


def visualize_labels(image, labels, threshold):
    for i in range(len(labels["labels"])):
        if labels["scores"][i] < threshold:
            continue
        if labels["labels"][i] == 1:
            image = cv2.rectangle(image, (int(labels['boxes'][i][0]), int(labels['boxes'][i][1]) ), ( int(labels['boxes'][i][2]),int(labels['boxes'][i][3])), (255,0,0), 1)
            image = cv2.putText(image, str(float(labels["scores"][i])), (int(labels['boxes'][i][0]), int(labels['boxes'][i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA, False)
    return image


def get_patch_test(patch, image, label, model, device, loss, threshold = 0.0):
    image = data_utils.image_to_tensor(image).to(device)
    attacked_image = image.clone()

    for l in label:
        attacked_image = am.insert_patch(attacked_image, patch, l, 0.3, device) 

    with torch.no_grad():
        prediction = model([image, attacked_image])

    image_before = visualize_labels(data_utils.tensor_to_image(image), prediction[0], threshold)
    image_after = visualize_labels(data_utils.tensor_to_image(attacked_image), prediction[1], threshold)

    return image_before, image_after
