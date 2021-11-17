import sys
sys.path.insert(1, '../mmdetection')

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import torch

def Check():
    # Check Pytorch installation
    import torch, torchvision
    print(torch.__version__, torch.cuda.is_available())

    # Check MMDetection installation
    import mmdet
    print(mmdet.__version__)

    # Check mmcv installation
    from mmcv.ops import get_compiling_cuda_version, get_compiler_version
    print(get_compiling_cuda_version())
    print(get_compiler_version())


ConfigFiles = {  'faster_rcnn':'../mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', }
CheckpointFiles = { 'faster_rcnn':'../mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth', }

def InitDetector(config, checkpoint):
    return init_detector(config, checkpoint, device = 'cuda:0' if torch.cuda.is_available() else 'cpu')