import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import attack_construction.attack_methods as attack
from torch.utils.tensorboard import SummaryWriter
from attack_construction.utils import save_patch_tensor


class Attack_module(nn.Module):
    def __init__(self, models, patch, device):
        super().__init__()
        self.models = models
        self.patch = patch
        self.device = device


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
                print(losses)
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



        
