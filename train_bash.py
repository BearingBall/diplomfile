import sys
sys.path.append('../')
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.hub

import torchvision
import torchvision.utils

from data import dataset as data
import data.utils as data_utils
import attack_construction.attack_methods as attack_methods
import attack_construction.metrics as metrics
import attack_construction.utils as attack_utils

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

print(torch.__version__)
print(torch.cuda_version)
print(torchvision.__version__)

parser = argparse.ArgumentParser(description = 'Train patch')
parser.add_argument('--data', help='data folder path', required=False, default='../train2017/train2017')
parser.add_argument('--labels', help='labels file path', required=False, default='../annotations_trainval2017/annotations/instances_train2017.json')
parser.add_argument('--val_data', help='validation folder path', required=False, default='../val2017/val2017')
parser.add_argument('--val_labels', help='validation labels file path', required=False, default='../annotations_trainval2017/annotations/instances_val2017.json')
parser.add_argument('--val_part', help='percentage of validation data', required=False, default=0.001)
parser.add_argument('--batch', help='batch size', required=False, default=1)
parser.add_argument('--val_batch', help='validation batch size', required=False, default=1)
parser.add_argument('--epoch', help='number of epoches', required=False, default=1)
parser.add_argument('--rate', help='graduation rate', required=False, default=0.03)
parser.add_argument('--device', help='0 - cpu, 1 - cuda', required=False, default=0)
parser.add_argument('--patch_name', help='patch name', required=False, default='patch_')

def main():
    args = parser.parse_args()
    print(args)

    folder_images = args.data
    val_images = args.val_data
    folder_labels = args.labels
    val_labels = args.val_labels
    device = torch.device("cpu") if args.device == 0 else torch.device("cuda:0")
    batch_size = args.batch
    val_batch = args.val_batch
    grad_rate = args.rate
    epoches = args.epoch
    patch_name = args.patch_name
    val_pecentage = args.val_part


    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    model.eval()
    model = model.float().to(device)

    for param in model.parameters():
        param.requires_grad = False 

    dataset = data.AdversarialDataset((640,640), folder_images, folder_labels) # todo: use resize to pull picture in batch
    dataset_val = data.AdversarialDataset((640,640), val_images, val_labels)

    train_loader  = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.Subset(dataset_val, range(0, int(len(dataset_val) * val_pecentage))), batch_size=val_batch, shuffle=False)

    patch = attack_methods.generate_random_patch()
    patch = patch.to(device)
    patch.requires_grad = True

    augmentations = torchvision.transforms.Compose([
        torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.2, hue=0.05),
        torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 0.3)),
        torchvision.transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
        torchvision.transforms.RandomRotation(degrees=(-30, 30)),])
    augmentations = None

    variation_coef = pow(10, -11)

    def loss_function(predict, patch, device):
        return metrics.general_objectness(predict, device) + variation_coef * metrics.total_variation(patch)

    for epoch in range(epoches):
        image_counter = 0
        eval_counter = 0

        for images, labels in train_loader:
            image_counter += batch_size
            loss, patch = attack_methods.training_step(model, patch, augmentations, images, labels, loss_function, device, grad_rate)
            print(f"ep:{epoch}, epoch_progress:{image_counter/len(dataset)}, batch_loss:{loss}")
            writer.add_scalar("Loss/train", loss, image_counter + epoch *  len(dataset))

            if (eval_counter == 0):
                eval_counter = 5
                attack_utils.save_tensor(patch.cpu(), './' + patch_name, True)
                loss_before, loss_after, mAP = attack_methods.validate(model, patch, augmentations, val_loader, loss_function, device)
                print(f"patch saved. VAL: raw loss:{loss_before}, attacked:{loss_after}, mAP:{mAP}")
                writer.add_scalar("Loss/val", loss_after, image_counter + epoch * len(dataset))
                writer.add_scalar("mAP/val", mAP, image_counter + epoch * len(dataset))
                writer.flush()

            eval_counter -= 1

    attack_utils.save_tensor(patch.cpu(), './' + patch_name, True)
    writer.close()


if __name__=='__main__':
    main()