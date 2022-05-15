import argparse


def parse_command_line_args_train():
    parser = argparse.ArgumentParser(description='Train patch')
    parser.add_argument(
        '--train_data',
        help='data folder path',
        required=False,
        type=str,
        default='../train2017/train2017',
    )
    parser.add_argument(
        '--train_labels',
        help='labels file path',
        required=False,
        type=str,
        default='../annotations_trainval2017/annotations/instances_train2017.json',
    )
    parser.add_argument(
        '--val_data',
        help='validation folder path',
        required=False,
        type=str,
        default='../val2017/val2017',
    )
    parser.add_argument(
        '--val_labels',
        help='validation labels file path',
        required=False,
        type=str,
        default='../annotations_trainval2017/annotations/instances_val2017.json',
    )
    parser.add_argument(
        '--val_part',
        help='percentage of validation data',
        required=False,
        type=float,
        default=1,
    )
    parser.add_argument(
        '--batch_size',
        help='batch size',
        required=False,
        type=int,
        default=1,
    )
    parser.add_argument(
        '--epochs',
        help='number of epochs',
        required=False,
        type=int,
        default=1,
    )
    parser.add_argument(
        '--rate',
        help='graduation rate',
        type=float,
        required=False,
        default=0.03,
    )
    parser.add_argument(
        '--device',
        help='0 - cpu, 1 - cuda',
        required=False,
        type=int,
        default=0,
    )
    parser.add_argument(
        '--experiment_dir',
        help='path to experiment folder',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--tv_scale',
        help='coefficient for total variation loss',
        required=False,
        type=float,
        default=1e-11,
    )
    parser.add_argument(
        '--step_save_frequency',
        help='save patch and validation performing frequency',
        required=False,
        default=100,
    )
    return parser.parse_args()


def parse_command_line_args_filter():
    parser = argparse.ArgumentParser(description='Annotation filter')
    parser.add_argument(
        '--input_path',
        help='path to annotation file',
        required=False,
        type=str,
        default='../annotations_trainval2017/annotations/instances_val2017.json',
    )
    parser.add_argument(
        '--output_path',
        help='path to result annotation file',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--is_crowd_include',
        help='include annotations with is_crowd = True',
        required=False,
        type=str,
        default=False,
    )
    parser.add_argument(
        '--min_bbox_area_ratio',
        help='min bbox_area_ratio included',
        required=False,
        type=str,
        default=0.05,
    )
    parser.add_argument(
        '--category',
        help='what type of object will be filtered',
        required=False,
        type=str,
        default=1,
    )
    return parser.parse_args()

def parse_command_line_args_validate():
    parser = argparse.ArgumentParser(description='Validate patch')
    parser.add_argument(
        '--patch',
        help='patch (png) path',
        required=False,
        type=str,
        default='patch.png',
    )
    parser.add_argument(
        '--val_data',
        help='validation folder path',
        required=False,
        type=str,
        default='../val2017/val2017',
    )
    parser.add_argument(
        '--val_labels',
        help='validation labels file path',
        required=False,
        type=str,
        default='../annotations_trainval2017/annotations/instances_val2017.json',
    )
    return parser.parse_args()