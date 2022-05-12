    # environment variables
dist='torch.distributed.launch'
python='torchrun'
executable='train_bash.py'

devices='0'

BATCH_SIZE=10
TV_SCALE=0.001

PATH_TO_COCO='/home/goncharenko/4tb_folder_first/data/coco_orig/'
PATH_TO_COCO_ANN=$PATH_TO_COCO'annotations_trainval2017/annotations/'


for i in 1
do
    CUDA_VISIBLE_DEVICES=$devices $python $executable \
        --val_labels '../annotation_cutted' \
        --val_part 1 \
        --batch_size $BATCH_SIZE \
        --epochs 20 \
        --device 1 \
	    --experiment_dir 'gtrain_12_05_2022' \
	    --tv_scale $TV_SCALE \
	
done

