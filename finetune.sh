#!/bin/bash

cd /home/svu/e0703350/Documents/projects/yoga_proj/yoga-project/

logf="logs/finetune_evolve.log"
CUDA_VISIBLE_DEVICE=1 nohup python evolve_train.py -r finetune_weight/checkpoint-epoch39.pth -s > ${logf} 2>&1 &
