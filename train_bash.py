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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from attack_construction.attack_class import Attack_class
from attack_construction.attack_class import train
from attack_construction.attack_class import validate
import attack_construction.attack_methods as attack_methods
from argument_parsing import parse_command_line_args_train
from attack_construction.attack_methods import adversarial_loss_function_batch
from attack_construction.utils import save_patch_tensor
from data import dataset as data
from RAdam.radam import RAdam
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import os

print(torch.__version__)
# This line doesnt work for me
# print(torch.cuda_version)
print(torchvision.__version__)
local_rank = int(os.environ["LOCAL_RANK"])
print(local_rank, " rank launched")
dist.init_process_group(backend="nccl")

def main():
    args = parse_command_line_args_train()

    train_images = args.train_data
    val_images = args.val_data
    train_labels = args.train_labels
    val_labels = args.val_labels
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
        model = model.float().to(f'cuda:{local_rank}')

        for param in model.parameters():
            param.requires_grad = False

    # TODO: use resize to pull picture in batch
    dataset = data.MsCocoDataset((640, 640), train_images, train_labels)
    dataset_val = data.MsCocoDataset((640, 640), val_images, val_labels)

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        num_workers=10,
        sampler=DistributedSampler(
                dataset=dataset),
    )

    small_train_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.Subset(dataset, range(0, 40)),
        batch_size=batch_size,
        num_workers=10,
        sampler=DistributedSampler(
                dataset=torch.utils.data.Subset(dataset, range(0, 40)))
    )

    small_val_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.Subset(dataset_val, range(0, int(len(dataset_val) * 0.01))),
        batch_size=30,
        num_workers=10
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.Subset(dataset_val, range(0, 10)),
        batch_size=30,
        num_workers=10,
    )

    annotation_file="../annotations_trainval2017/annotations/instances_val2017.json"

    patch = attack_methods.generate_random_patch()
    patch = patch.to(f'cuda:{local_rank}')
    patch.requires_grad = True

    optimizer = RAdam([patch], lr=grad_rate)

    augmentations = torchvision.transforms.Compose([
        torchvision.transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2)),
        torchvision.transforms.GaussianBlur(kernel_size=(5, 5), sigma=(1.2, 1.2)),
        torchvision.transforms.RandomRotation(degrees=(-5, 5)),
    ])

    augmentations = None

    loss_function = partial(adversarial_loss_function_batch, tv_scale=args.tv_scale, local_rank=local_rank)

    attack_module = Attack_class(models, patch, local_rank)
    attack_module = DDP(attack_module, device_ids=[local_rank], output_device=local_rank)

    writer = SummaryWriter(log_dir=experiment_dir.as_posix())

    for epoch in range(epoches):
        train(attack_module, small_train_loader, augmentations, optimizer, writer, loss_function)
        mAPs = validate(attack_module, val_loader, augmentations, annotation_file, local_rank)

        if (local_rank == 0):
            print("mAPs: ", mAPs)
            for i, mAP in enumerate(mAPs):
                writer.add_scalar('mAP, model: ' + str(i), mAP, epoch)

            save_patch_tensor(attack_module.module.patch, experiment_dir, epoch=epoch, step=0, save_mode='both')

        dist.barrier()

main()