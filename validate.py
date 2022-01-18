import sys
sys.path.append('../')

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.hub
import torchvision
import torchvision.utils

import attack_construction.attack_methods as attack_methods
from attack_construction.utils import load_tensor_from_image
from argument_parsing import parse_command_line_args_validate
from data import dataset as data

print(torch.__version__)
print(torchvision.__version__)


def main():
    args = parse_command_line_args_validate()

    patch_file = args.patch
    val_images = args.val_data
    val_labels = args.val_labels
    device = torch.device("cuda:0")

    # need for good experiment logging creation

    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    model.eval()
    model = model.float().to(device)

    for param in model.parameters():
        param.requires_grad = False

    # TODO: use resize to pull picture in batch
    dataset_val = data.AdversarialDataset((640, 640), val_images, val_labels)

    val_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.Subset(dataset_val, range(0, int(len(dataset_val)))),
        batch_size=30,
        shuffle=True,
        num_workers=10
    )

    patch = load_tensor_from_image(patch_file)
    patch = patch.to(device)

    obj, tv, mAP = attack_methods.validate(
        model, 
        patch, 
        None, 
        val_loader, 
        device, 
        val_labels, 
        None)

    print(f'Validation closed. VAL: objectness:{obj}, attacked:{tv}, mAP:{mAP}')


if __name__ == '__main__':
    main()
