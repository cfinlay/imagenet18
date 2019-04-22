#!/bin/bash

TIMESTAMP=`date +%y-%m-%dT%H%M%S`  # Use this in LOGDIR

ulimit -n 4096

python -m torch.distributed.launch \
  --nproc_per_node=4 --nnodes=1 --node_rank=0 \
  ./train_imagenet_nv.py /mnt/data/scratch/ImageNet \
  --workers=4 \
  --init-bn0 \
  --logdir ./runs/run-$TIMESTAMP --distributed  \
  --phases "[{'ep': 0, 'sz': 128, 'bs': 256, 'trndir': '-sz/160'}, {'ep': (0, 8), 'lr': (0.5, 1.0)}, {'ep': (8, 15), 'lr': (1.0, 0.125)}, {'ep': 15, 'sz': 224, 'bs': 112, 'trndir': '-sz/320', 'min_scale': 0.087}, {'ep': (15, 25), 'lr': (0.22, 0.022)}, {'ep': (25, 28), 'lr': (0.022, 0.0022)}, {'ep': 28, 'sz': 288, 'bs': 64, 'min_scale': 0.5, 'rect_val': True}, {'ep': (28, 29), 'lr': (0.00125, 0.000125)}]" --skip-auto-shutdown



