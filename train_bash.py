import sys
sys.path.append('../')

import warnings
warnings.filterwarnings("ignore")

from functools import partial
from pathlib import Path

import torch
import torch.hub
import torchvision
import torchvision.utils
from torch.utils.tensorboard import SummaryWriter

import attack_construction.attack_methods as attack_methods
from attack_construction.utils import save_patch_tensor
from argument_parsing import parse_command_line_args
from attack_construction.attack_methods import adversarial_loss_function
from data import dataset as data

print(torch.__version__)
# This line doesnt work for me
# print(torch.cuda_version)
print(torchvision.__version__)


def main():
    args = parse_command_line_args()

    train_images = args.train_data
    val_images = args.val_data
    train_labels = args.train_labels
    val_labels = args.val_labels
    device = torch.device("cpu") if args.device == 0 else torch.device("cuda:0")
    batch_size = args.batch_size
    grad_rate = args.rate
    epoches = args.epochs
    experiment_dir = Path(args.experiment_dir)
    val_pecentage = args.val_part
    step_save_frequency = args.step_save_frequency

    # need for good experiment logging creation
    experiment_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=experiment_dir.as_posix())

    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    model.eval()
    model = model.float().to(device)

    for param in model.parameters():
        param.requires_grad = False

    # TODO: use resize to pull picture in batch
    dataset = data.AdversarialDataset(None, train_images, train_labels)
    dataset_val = data.AdversarialDataset(None, val_images, val_labels)

    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    small_val_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.Subset(dataset_val, range(0, int(len(dataset_val) * val_pecentage))),
        batch_size=batch_size,
        shuffle=False,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.Subset(dataset_val, range(0, int(len(dataset_val) * val_pecentage))),
        batch_size=batch_size,
        shuffle=True,
    )

    patch = attack_methods.generate_random_patch()
    patch = patch.to(device)
    patch.requires_grad = True

    augmentations = torchvision.transforms.Compose([
        torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.2, hue=0.05),
        torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 0.3)),
        torchvision.transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
        torchvision.transforms.RandomRotation(degrees=(-30, 30)),
    ])

    loss_function = partial(adversarial_loss_function, tv_scale=args.tv_scale)

    for epoch in range(epoches):
        image_counter = 0
        for step_num, (images, labels, _) in enumerate(train_loader):
            image_counter += batch_size
            loss, patch = attack_methods.training_step(
                model=model,
                patch=patch,
                augmentations=None,
                images=images,
                labels=labels,
                loss=loss_function,
                device=device,
                grad_rate=grad_rate,
            )
            # TODO: apply tqdm library for progress logging
            print(f'ep:{epoch}, epoch_progress:{image_counter/len(dataset)}, batch_loss:{loss}')
            writer.add_scalar('Loss/train', loss, step_num)

            if step_num % step_save_frequency == 0:
                save_patch_tensor(patch, experiment_dir, epoch=epoch, step=step_num, save_mode='both')
                obj, tv, mAP = attack_methods.validate(model, patch, augmentations, small_val_loader, device, val_labels)
                print(f'patch saved. VAL: objectness:{obj}, attacked:{tv}, mAP:{mAP}')
                writer.add_scalar('Loss/val_obj', obj, step_num)
                writer.add_scalar('Loss/val_tv', tv, step_num)
                writer.add_scalar('mAP/val', mAP, step_num)
                writer.flush()

        # at least one time in epoch you need full validation
        save_patch_tensor(patch, experiment_dir, epoch=epoch, step=step_num)
        obj, tv, mAP = attack_methods.validate(model, patch, augmentations, val_loader, device, val_labels)
        print(f'patch saved. VAL: objectness:{obj}, attacked:{tv}, mAP:{mAP}')
        writer.add_scalar('Loss/val_obj', obj, epoch)
        writer.add_scalar('Loss/val_tv', tv, epoch)
        writer.add_scalar('mAP/val', mAP, epoch)
        writer.flush()

    writer.close()


if __name__ == '__main__':
    main()
