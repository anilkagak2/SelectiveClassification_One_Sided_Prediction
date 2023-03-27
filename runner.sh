
#model_name='ShuffleNetV2'
#model_name='ResNet18'
model_name='ResNet32'
#model_name='ResNet50'

dataset='cifar10'
data_path='/home/anilkag/code/data/cifar/'
#data_path='/home/anil/github/datasets/cifar/'
#data_path='/projectnb/scaffold/anilkag/datasets/cifar/'

ckpt='"'


CUDA_VISIBLE_DEVICES='0' python train_osp.py --dataset $dataset --data_path $data_path --model_name $model_name --method 'our-SD' \
	--epochs 200 --lr 0.01 --max_lr 0.0001 --wd 0.0005 --batch_size 128 --eval_batch_size 256 --ema-decay 0.996 \
	--_ckpt $ckpt \
        --coverage_list 1. 0.95 0.90 \
        --error_list 0.005 0.01 0.02 \
        --mus_list 0.49 1.67 #--eval 

#        --mus_list 0.49 1.67 --n_threshold 1000 --eval 
#        --mus_list 0.8 --n_threshold 1000 #--eval 


