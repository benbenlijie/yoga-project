#!/bin/bash

cd /home/svu/e0703350/Documents/projects/yoga_proj/yoga-project/

logf="test_yoga_score.log"
CUDA_VISIBLE_DEVICE=2 nohup python test_yoga_score.py -r /hpctmp/e0703350/projects/yoga-saved/models/Yoga_Score_Net_filtered_data2_1dconv_arc_16-32-32-32-k15_noeasy/1031_093746/model_best.pth > ${logf} 2>&1 &
