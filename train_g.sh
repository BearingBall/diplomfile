    # environment variables
dist='torch.distributed.launch'
python='torchrun'
executable='train_bash.py'

devices='0,1'

BATCH_SIZE=7
TV_SCALE=0.001

PATH_TO_COCO='/home/goncharenko/4tb_folder_first/data/coco_orig/'
PATH_TO_COCO_ANN=$PATH_TO_COCO'annotations_trainval2017/annotations/'

NUM_TRAINERS=2
master_port=1234

for i in 1
do
    CUDA_VISIBLE_DEVICES=$devices $python --nproc_per_node=$NUM_TRAINERS --master_port=$master_port $executable \
        --val_part 1 \
        --batch_size $BATCH_SIZE \
        --epochs 20 \
        --device 1 \
        --rate 0.12 \
	    --experiment_dir 'gtrain_17_05_2022' \
	    --tv_scale $TV_SCALE \
	
done

