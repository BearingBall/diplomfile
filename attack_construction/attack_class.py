#from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import attack_construction.attack_methods as attack
from torch.utils.tensorboard import SummaryWriter
from attack_construction.utils import save_patch_tensor
from progress.bar import IncrementalBar
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def train(my_complex_model, train_dataloader, augmentations, optimizer, writer, loss):
    bar = IncrementalBar('Epoch progress', max = len(train_dataloader))

    for step_num, (images, labels, _, _) in enumerate(train_dataloader):
        for model_index in range(len(my_complex_model.models)):
            print("predicting")
            prediction = my_complex_model(images, labels, model_index, augmentations)
            costs = loss(prediction, my_complex_model.patch)
            cost = sum(costs)
            print(costs)
            cost.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            my_complex_model.patch.data.clamp_(0,1)

        bar.next()
    
    bar.finish()


#в разработке
def validate(my_complex_model, val_dataloader, augmentations, annotation_file):
    mAPs = []

    for model_index in range(len(my_complex_model.models)):
        annotation_after = []

        for val_idx, (images, labels, img_ids, scale_factor) in enumerate(val_dataloader):
            with torch.no_grad():
                prediction = my_complex_model(images, labels, model_index, augmentations)

            for i in range(len(prediction)):
                for j in range(len(prediction[i]["labels"])):
                    annotation_after.append({
                        'image_id': img_ids[i].item(),
                        'category_id': prediction[i]["labels"][j].item(),
                        'bbox': [
                          prediction[i]["boxes"][j][0].item() / scale_factor[i][0].item(),
                          prediction[i]["boxes"][j][1].item() / scale_factor[i][1].item(),
                          (prediction[i]["boxes"][j][2].item() - prediction[i]["boxes"][j][0].item()) / scale_factor[i][0].item(),
                          (prediction[i]["boxes"][j][3].item() - prediction[i]["boxes"][j][1].item()) / scale_factor[i][1].item()
                    ],
                    "score": prediction[i]['scores'][j].item()
                })

        with open("tmp.json", 'w') as f_after:
            json.dump(annotation_after, f_after)

        cocoGt = COCO(annotation_file)
        cocoDt = cocoGt.loadRes("./tmp.json")

        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = cocoGt.getImgIds()
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        mAPs.append(np.mean(cocoEval.stats))
    
    return mAPs 




class Attack_class(nn.Module):
    def __init__(self, models, patch):
        super().__init__()
        self.models = models
        self.patch = nn.Parameter(data=patch)


    def forward(self, images, labels, model_index, augmentations):
        if (model_index >= len(self.models)):
            return None

        augmented_patch = self.patch if augmentations is None else augmentations(self.patch)

        attacked_images = []

        for i, image in enumerate(images):
            attacked_image = image.cuda()

            if labels[i][0][2] * labels[i][0][3] != 0:
                for label in labels[i]:
                    attacked_image = attack.insert_patch(attacked_image, augmented_patch, label, 0.4, True)

            attacked_images.append(attacked_image)

        if (len(attacked_images) == 0):
            return None

        return self.models[model_index](attacked_images)









    #ниже устаревшее
'''
    def train(self, epochs, train_loader, batch_size, augmentations, loss_function, optimizer, experiment_dir, step_save_frequency, val_loader, small_val_loader, val_labels):
        writer = SummaryWriter(log_dir=experiment_dir.as_posix())
        for epoch in range(epochs):
            image_counter = 0
            prev_steps = epoch * len(train_loader)

            for step_num, (images, labels, _, _) in enumerate(train_loader):
                image_counter += batch_size

                losses = []
                for model in self.models:
                    loss, patch = attack.training_step(
                        model=model,
                        patch=self.patch,
                        augmentations=augmentations,
                        images=images,
                        labels=labels,
                        loss=loss_function,
                        device=self.device,
                        optimizer = optimizer,
                    )
                    losses.append(loss)
                    self.patch = patch
                loss = np.mean(losses)
                
                # TODO: apply tqdm library for progress logging
                print(f"ep:{epoch}, epoch_progress:{step_num/len(train_loader)}, batch_loss:{loss}")
                writer.add_scalar('Loss/train', loss, step_num + prev_steps)

                if step_num % step_save_frequency == 0:
                    self.log_results(writer, experiment_dir, epoch, step_num, prev_steps, small_val_loader, val_labels, augmentations)

            # at least one time in epoch you need full validation
            self.log_results(writer, experiment_dir, epoch, step_num, prev_steps, val_loader, val_labels, augmentations)

        writer.close()


    def validate(self, validate_dir, val_loader, val_labels, augmentations):
        objs = 0
        tvs = 0
        maps = 0
        for model in self.models:
            obj, tv, mAP = attack.validate(
                model, 
                self.patch, 
                augmentations, 
                val_loader, 
                self.device, 
                val_labels, 
                validate_dir)
            objs += obj
            tvs += tv
            maps += mAP

        obj = objs / len(self.models)
        tv = tvs / len(self.models)
        mAP = maps / len(self.models)

        print(f'Patch validated. VAL: objectness:{obj}, tv:{tv}, mAP:{mAP}')
        return obj, tv, mAP
        

    def log_results(self, writer, experiment_dir, epoch, step_num, prev_steps, small_val_loader, val_labels, augmentations):
        save_patch_tensor(self.patch, experiment_dir, epoch=epoch, step=step_num, save_mode='both')
        validate_dir = experiment_dir / ('validate_epoch_' + str(epoch) + '_step_' + str(step_num))
        validate_dir.mkdir(parents=True, exist_ok=True)
        obj, tv, mAP = self.validate(
                validate_dir,
                small_val_loader, 
                val_labels, 
                augmentations)
        print(f'patch saved. VAL: objectness:{obj}, attacked:{tv}, mAP:{mAP}')
        writer.add_scalar('Loss/val_obj', obj, step_num + prev_steps)
        writer.add_scalar('Loss/val_tv', tv, step_num + prev_steps)
        writer.add_scalar('mAP/val', mAP, step_num + prev_steps)
        writer.flush()


'''
        
