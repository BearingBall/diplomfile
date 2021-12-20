import argparse


def parse_command_line_args():
    parser = argparse.ArgumentParser(description='Train patch')
    parser.add_argument('--data', help='data folder path', required=False, default='../train2017/train2017')
    parser.add_argument('--labels', help='labels file path', required=False,
                        default='../annotations_trainval2017/annotations/instances_train2017.json')
    parser.add_argument('--val_data', help='validation folder path', required=False, default='../val2017/val2017')
    parser.add_argument('--val_labels', help='validation labels file path', required=False,
                        default='../annotations_trainval2017/annotations/instances_val2017.json')
    parser.add_argument('--val_part', help='percentage of validation data', required=False, default=0.001)
    parser.add_argument('--batch', help='batch size', required=False, default=1)
    parser.add_argument('--val_batch', help='validation batch size', required=False, default=1)
    parser.add_argument('--epoch', help='number of epoches', required=False, default=1)
    parser.add_argument('--rate', help='graduation rate', required=False, default=0.03)
    parser.add_argument('--device', help='0 - cpu, 1 - cuda', required=False, default=0)
    parser.add_argument('--patch_name', help='patch name', required=False, default='patch_')
    return parser.parse_args()
