configfile=$1
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --master_port 5689 train.py --config $configfile --tag chinese