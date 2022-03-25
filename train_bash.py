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

import attack_construction.attack_class as attack_class
import attack_construction.attack_methods as attack_methods
from argument_parsing import parse_command_line_args_train
from attack_construction.attack_methods import adversarial_loss_function_batch
from data import dataset as data
from RAdam.radam import RAdam

print(torch.__version__)
# This line doesnt work for me
# print(torch.cuda_version)
print(torchvision.__version__)


def main():
    args = parse_command_line_args_train()

    train_images = args.train_data
    val_images = args.val_data
    train_labels = args.train_labels
    val_labels = args.val_labels
    device = torch.device("cpu") if int(args.device) == 0 else torch.device("cuda:0")
    batch_size = args.batch_size
    grad_rate = args.rate
    epoches = args.epochs
    experiment_dir = Path(args.experiment_dir)
    val_pecentage = args.val_part
    step_save_frequency = int(args.step_save_frequency)

    # need for good experiment logging creation
    experiment_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=experiment_dir.as_posix())

    models = [  torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True),
                torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True),
                torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
             ]

    for model in models:
        model.eval()
        model = model.float().to(device)

        for param in model.parameters():
            param.requires_grad = False

    # TODO: use resize to pull picture in batch
    dataset = data.MsCocoDataset((640, 640), train_images, train_labels)
    dataset_val = data.MsCocoDataset((640, 640), val_images, val_labels)

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=10
    )

    small_val_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.Subset(dataset_val, range(0, int(len(dataset_val) * val_pecentage))),
        batch_size=30,
        shuffle=False,
        num_workers=10
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.Subset(dataset_val, range(0, int(len(dataset_val) * val_pecentage))),
        batch_size=30,
        shuffle=True,
        num_workers=10
    )

    patch = attack_methods.generate_random_patch()
    patch = patch.to(device)
    patch.requires_grad = True

    optimizer = RAdam([patch], lr=grad_rate)

    augmentations = torchvision.transforms.Compose([
        torchvision.transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2)),
        torchvision.transforms.GaussianBlur(kernel_size=(5, 5), sigma=(1.2, 1.2)),
        torchvision.transforms.RandomRotation(degrees=(-5, 5)),
    ])

    augmentations = None

    loss_function = partial(adversarial_loss_function_batch, tv_scale=args.tv_scale)

    attack_module = attack_class.Attack_module(models, patch, device)

    attack_module.train(epochs=epoches, 
                        train_loader=train_loader,
                        batch_size=batch_size,
                        augmentations=augmentations, 
                        loss_function=loss_function, 
                        optimizer=optimizer, 
                        experiment_dir=experiment_dir, 
                        step_save_frequency=step_save_frequency, 
                        val_loader=val_loader, 
                        small_val_loader=small_val_loader, 
                        val_labels=val_labels)

if __name__ == '__main__':
    main()
