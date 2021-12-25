    # environment variables
dist='torch.distributed.launch'
python='/home/goncharenko/4tb_folder_first/goncharenko/students/2.yerkovich/.stepan_diploma/bin/python3.9'
executable='train_bash.py'

devices='0'

BATCH_SIZE=12
TV_SCALE=0.001

PATH_TO_COCO='/home/goncharenko/4tb_folder_first/data/coco_orig/'
PATH_TO_COCO_ANN=$PATH_TO_COCO'annotations_trainval2017/annotations/'


for i in 1
do
    CUDA_VISIBLE_DEVICES=$devices $python $executable \
        --train_data $PATH_TO_COCO'train2017' \
        --train_labels $PATH_TO_COCO_ANN'instances_train2017.json' \
        --val_data $PATH_TO_COCO'val2017' \
        --val_labels $PATH_TO_COCO_ANN'instances_val2017.json' \
        --val_part 0.04 \
        --batch_size $BATCH_SIZE \
        --epochs 150 \
        --rate 3e-2\
        --device 1 \
	--experiment_dir 'experiments/TV_SCALE_'$TV_SCALE'/' \
	--tv_scale $TV_SCALE \
	
done

