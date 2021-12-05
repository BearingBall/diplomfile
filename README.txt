Wellcome to my sandbox

Kernell version 3.8.6 (but other can be possible mb)
It is needed torch 1.10.0+cu113 + torchvision 0.11.1+cu113 (and some other pip stuff).
Dataset: MSCOCO2017 (http://images.cocodataset.org/zips/train2017.zip + http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

1 click training:
trainBash.py datasetFolder labelfile 
For example:
trainBash.py "../train2017/train2017" "../annotations_trainval2017/annotations/instances_train2017.json"

jptr notebook train: trainTorch.ipynb
jptr notebook test: testTorch.ipynb

OTHER FILES ONLY FOR DEVELOPERS (dont touch without reasons)
Good luck!

