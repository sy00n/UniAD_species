export PYTHONPATH=../../:$PYTHONPATH
NUMBER_OF_GPUS_PER_NODE=1

# python -m torch.distributed.launch --nproc_per_node=$1 ../../tools/train_val.py -e
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nnodes=1 --nproc_per_node=$NUMBER_OF_GPUS_PER_NODE \
		 ../../tools/train_val.py -e


