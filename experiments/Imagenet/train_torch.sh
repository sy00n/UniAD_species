export PYTHONPATH=../../:$PYTHONPATH

CUDA_VISIBLE_DEVICES=1,2,5,6 CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node=$1 ../../tools/train_val.py --config /home/workspace/UniAD/experiments/Imagenet/new_config.yaml