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

print(torch.__version__)
# This line doesnt work for me
# print(torch.cuda_version)
print(torchvision.__version__)


def main(rank, world_size):
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
        model = model.float().cuda()

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

    small_train_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.Subset(dataset, range(0, int(len(dataset) * 0.001))),
        batch_size=batch_size,
        shuffle=True,
        num_workers=10
    )

    small_val_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.Subset(dataset_val, range(0, int(len(dataset_val) * 0.01))),
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

    annotation_file="../annotations_trainval2017/annotations/instances_val2017.json"

    patch = attack_methods.generate_random_patch()
    patch = patch.cuda()
    patch.requires_grad = True

    optimizer = RAdam([patch], lr=grad_rate)

    augmentations = torchvision.transforms.Compose([
        torchvision.transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2)),
        torchvision.transforms.GaussianBlur(kernel_size=(5, 5), sigma=(1.2, 1.2)),
        torchvision.transforms.RandomRotation(degrees=(-5, 5)),
    ])

    augmentations = None

    loss_function = partial(adversarial_loss_function_batch, tv_scale=args.tv_scale)

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    attack_module = Attack_class(models, patch)
    attack_module = DDP(attack_module)

    writer = SummaryWriter(log_dir=experiment_dir.as_posix())

    for epoch in range(epoches):
        train(attack_module, train_loader, augmentations, optimizer, writer, loss_function)
        mAPs = validate(attack_module, val_loader, augmentations, annotation_file)
        print("mAPs: ", mAPs)
        for i, mAP in enumerate(mAPs):
            writer.add_scalar('mAP, model: ' + str(i), mAP, epoch)

        save_patch_tensor(attack_module.patch, experiment_dir, epoch=epoch, step=0, save_mode='both')


if __name__ == '__main__':
    world_size = 2
    mp.spawn(main,
        args=(world_size,),
        nprocs=world_size,
        join=True)

from torch.nn.parallel import DistributedDataParallel as DDP