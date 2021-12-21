import torch
import torch
import torchvision.transforms as T
import numpy as np
import data.utils as data_utils
import attack_construction.metrics as attack_metric
import random
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def generate_random_patch(resolution = (70,70)):
    return torch.rand(3, resolution[0], resolution[1])


def insert_patch(image, patch, box, ratio, device, random_place = False):
    patch_size=(int(box[3] * ratio), int(box[2] * ratio))

    if patch_size[0] == 0 or patch_size[1] == 0: # cant insert patch with this box parameters
        return image

    resized_patch = T.Resize(size = patch_size)(patch)

    patch_x_offset = box[2] * (random.uniform(0, 1-ratio) if random_place else 0.5-ratio/2)
    patch_y_offset = box[3] * (random.uniform(0, 1-ratio) if random_place else 0.5-ratio/2)
    x_shift = int(box[0] + patch_x_offset)
    y_shift = int(box[1] + patch_y_offset)

    padding = (x_shift, y_shift, image.shape[2] - x_shift - patch_size[1], image.shape[1] - y_shift - patch_size[0])
    padded_patch = T.Pad(padding=padding)(resized_patch)
    patch_mask = T.Pad(padding=padding)(torch.ones(size=(3, patch_size[0], patch_size[1]))).to(device)
    ones = torch.ones_like(image).to(device)
    result = (torch.ones_like(image).to(device) - patch_mask.to(device)) * image + padded_patch
    patch_mask.detach()
    ones.detach()
    return  result

def training_step(model, patch, augmentations, images, labels, loss, device, grad_rate):
    torch.cuda.empty_cache()
    attacked_images = []

    for image in images:
        attacked_images.append(data_utils.image_to_tensor(image).to(device))

    #for attackedIm in attackedImage:
    #    attackedIm.requires_grad = True

    #with torch.no_grad():
    #    clear_predict = model(attacked_images)

    augmented_patch = patch if augmentations == None else augmentations(patch)

    for i in range(len(attacked_images)):
        for label in labels[i]:
            attacked_images[i] = insert_patch(attacked_images[i], augmented_patch, label, 0.3, device) 

    predict = model(attacked_images)

    costs = []

    for i in range(len(predict)):
        cost = loss(predict[i], patch, device)
        if cost < 0.1:
            continue
        try:
            grad = torch.autograd.grad(cost, patch, retain_graph=False, create_graph=False,  allow_unused=True)[0]
            if grad != None:
                patch = patch - grad_rate*grad.sign()
        except:
             pass
        costs.append(cost.detach().cpu())

    for attacked_image in attacked_images:
        attacked_image.detach()
    
    return np.mean(np.asarray(costs)), patch

def validate(model, patch, augmentations, val_loader, device, annotation_file = "../../annotations_trainval2017/annotations/instances_val2017.json"):
    torch.cuda.empty_cache()
    
    objectness = []
    annotation_after = []

    for images, labels, img_ids in val_loader:
        attacked_images = []

        for image in images:
            attacked_images.append(data_utils.image_to_tensor(image).to(device))

        with torch.no_grad():         
            clear_predict = model(attacked_images)

        augmented_patch = patch if augmentations == None else augmentations(patch)

        for i in range(len(attacked_images)):
            for label in labels[i]:
                attacked_images[i] = insert_patch(attacked_images[i], augmented_patch, label, 0.3, device) 

        with torch.no_grad():        
            predict = model(attacked_images)

        for i in range(len(predict)):
            for j in range(len(predict[i]["labels"])):
                annotation_after.append({
                    'image_id' : img_ids[i].item(),
                    'category_id': predict[i]["labels"][j].item(),
                    'bbox': [
                        predict[i]["boxes"][j][0].item(),
                        predict[i]["boxes"][j][1].item(),
                        predict[i]["boxes"][j][2].item(),
                        predict[i]["boxes"][j][3].item()
                    ],
                    "score": predict[i]['scores'][j].item()
                    })

        objectness = objectness + [attack_metric.general_objectness(predict[i], device).detach().cpu() for i in range(len(clear_predict))]

    with open("tmp.json", 'w') as f_after:
        json.dump(annotation_after, f_after)     

    val_loader.dataset
    cocoGt=COCO(annotation_file)
    cocoDt=cocoGt.loadRes("./tmp.json")

    cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
    cocoEval.params.imgIds = cocoGt.getImgIds()
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    total_variation = attack_metric.total_variation(patch).detach().cpu()
    return np.mean(np.asarray(objectness)), total_variation, np.mean(cocoEval.stats)

        

        