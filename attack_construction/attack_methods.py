from configparser import Interpolation
import json
import random

import numpy as np
import cv2 as cv
import torch
import torchvision.transforms as T
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import attack_construction.metrics as attack_metric
from attack_construction.utils import visualize_labels_predicted, visualize_labels_gt
import data.utils as data_utils
import attack_construction.metrics as metrics


def adversarial_loss_function_batch(predicts, patch, device, tv_scale):
    return [adversarial_loss_function(predict, patch, device, tv_scale) for predict in predicts]


def adversarial_loss_function(predict, patch, device, tv_scale):
    return metrics.general_objectness(predict, device) + tv_scale * metrics.total_variation(patch)


def generate_random_patch(resolution=(90, 90)):
    return torch.rand(3, resolution[0], resolution[1])


def insert_patch(image, patch, box, ratio, device, random_place=False):
    patch_size = (int(box[3] * ratio), int(box[2] * ratio))

    # cant insert patch with this box parameters
    if patch_size[0] == 0 or patch_size[1] == 0:
        return image

    resized_patch = T.Resize(size=patch_size, interpolation=T.InterpolationMode.BICUBIC)(patch)

    patch_x_offset = box[2] * (random.uniform(0, 1 - ratio) if random_place else 0.5 - ratio / 2)
    patch_y_offset = box[3] * (random.uniform(0.3, 1 - ratio) if random_place else 0.5 - ratio / 2)
    x_shift = int(box[0] + patch_x_offset)
    y_shift = int(box[1] + patch_y_offset)

    padding = (x_shift, y_shift, image.shape[2] - x_shift - patch_size[1], image.shape[1] - y_shift - patch_size[0])
    padded_patch = T.Pad(padding=padding)(resized_patch)
    patch_mask = T.Pad(padding=padding)(torch.ones(size=(3, patch_size[0], patch_size[1]))).to(device)
    result = padded_patch + (torch.ones_like(image) - patch_mask) * image
    return result


def training_step(model, patch, augmentations, images, labels, loss, device, optimizer):
   
    attacked_images = [] #torch.tensor(image.to(device), requires_grad = True) for image in images

    augmented_patch = patch if augmentations is None else augmentations(patch)

    for i, image in enumerate(images):
        if labels[i][0][2] * labels[i][0][3] != 0:
            attacked_image = image.to(device)

            for label in labels[i]:
                attacked_image = insert_patch(attacked_image, augmented_patch, label, 0.4, device, True)

            attacked_images.append(attacked_image)

    costMean = 0
    
    if len(attacked_images) != 0:

        predict = model(attacked_images)

        costs = loss(predict, patch, device)

        cost = sum(costs)
    
        cost.backward()

        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            patch.data.clamp_(0,1)

        costMean = np.mean(np.asarray([cost.detach().cpu().numpy() for cost in costs]))

    return costMean, patch


def validate(
        model,
        patch,
        augmentations,
        val_loader,
        device,
        annotation_file="../../annotations_trainval2017/annotations/instances_val2017.json",
        example_file=None
):
    model.eval()
    torch.cuda.empty_cache()

    example_batch_num = random.randint(0, len(val_loader))

    objectness = []
    annotation_after = []
    for val_idx, (images, labels, img_ids, scale_factor) in enumerate(val_loader):
        attacked_images = images.to(device)

        augmented_patch = patch if augmentations is None else augmentations(patch)

        if augmented_patch is not None:
            for i, _ in enumerate(images):
                for label in labels[i]:
                    attacked_images[i] = insert_patch(attacked_images[i], augmented_patch, label, 0.4, device, True)

        with torch.no_grad():
            predict = model(attacked_images)

        for i in range(len(predict)):
            for j in range(len(predict[i]["labels"])):
                annotation_after.append({
                    'image_id': img_ids[i].item(),
                    'category_id': predict[i]["labels"][j].item(),
                    'bbox': [
                        predict[i]["boxes"][j][0].item() / scale_factor[i][0].item(),
                        predict[i]["boxes"][j][1].item() / scale_factor[i][1].item(),
                        (predict[i]["boxes"][j][2].item() - predict[i]["boxes"][j][0].item()) / scale_factor[i][0].item(),
                        (predict[i]["boxes"][j][3].item() - predict[i]["boxes"][j][1].item()) / scale_factor[i][1].item()
                    ],
                    "score": predict[i]['scores'][j].item()
                })

        objectness.extend([
            attack_metric.general_objectness(single_image_predict, device).detach().cpu()
            for single_image_predict in predict
        ])

        if example_batch_num == val_idx and example_file is not None:
            for j, (image) in enumerate(images):
                cv.imwrite((example_file / (str(j) + "_gt.png")).as_posix(), cv.cvtColor(visualize_labels_gt(data_utils.tensor_to_image(image), labels[j]), cv.COLOR_RGB2BGR))
                cv.imwrite((example_file / (str(j) + "_predicted.png")).as_posix(), cv.cvtColor(visualize_labels_predicted(data_utils.tensor_to_image(attacked_images[j]), predict[j], 0.2), cv.COLOR_RGB2BGR))

    with open("tmp.json", 'w') as f_after:
        json.dump(annotation_after, f_after)

    cocoGt = COCO(annotation_file)
    cocoDt = cocoGt.loadRes("./tmp.json")

    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = cocoGt.getImgIds()
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    total_variation = 0

    if patch is not None:
        total_variation = attack_metric.total_variation(patch).detach().cpu()
        
    return np.mean(np.asarray(objectness)), total_variation, np.mean(cocoEval.stats)
